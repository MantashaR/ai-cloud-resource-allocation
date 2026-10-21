"""End-to-end experiment, explainable decisions and what-if simulation.

Data timeline (default 28 days):
  [ ---- train (14d) ---- | -- calibrate (7d) -- | -- TEST (7d, unseen) -- ]

1. Forecasters (CPU, memory, requests) are fitted on 'train' and checked on
   'calibrate' -> conformal safety offset; RL agent trained on 'calibrate'.
2. Forecasters re-fitted on train+calibrate, then frozen.
3. Every autoscaling policy runs on the unseen test week and is compared.
"""
import copy
import math
import numpy as np
import pandas as pd

from .simulator import (simulate, enrich, summarize, StaticPolicy, ReactivePolicy, response_time,
                        max_util_for_latency, STEPS_PER_DAY, STEP_MIN)
from .ai_models import (Forecaster, calibrated_forecast, PredictivePolicy, QLearningScaler,
                        allowed_targets, pods_needed, TREND_NAMES, VOL_NAMES, HOUR_NAMES, MEM_TARGET)
from .anomaly import detect_anomalies, evaluate_against_labels

HYBRID = QLearningScaler.name


def run_experiment(wl, pod_cpu=4, pod_mem=8, price_hr=0.20, boot_steps=2, max_rt=500,
                   avail_target=99.0, max_util=0.80, sla_weight=10.0, rl_episodes=40,
                   test_days=7, progress=None):
    cpu = wl["cpu"].to_numpy(float)
    mem = wl["memory"].to_numpy(float)
    req = wl["requests"].to_numpy(float)
    n = len(cpu)
    test_len = min(test_days * STEPS_PER_DAY, n // 4)
    split = n - test_len
    cal_start = split - test_len
    env_kwargs = dict(pod_cpu=pod_cpu, price_hr=price_hr, boot_steps=boot_steps)
    horizon = boot_steps + 1
    target_util = min(max_util, max_util_for_latency(max_rt))
    say = progress or (lambda *_: None)

    # ---- forecasters -------------------------------------------------------
    say("Training CPU forecaster ...", 0.03)
    cpu_cal_fc, fc_cpu, off_cpu = calibrated_forecast(cpu, cal_start, split, horizon)
    say("Training memory forecaster ...", 0.10)
    mem_cal_fc, fc_mem, off_mem = calibrated_forecast(mem, cal_start, split, horizon)
    say("Training request-rate forecaster ...", 0.17)
    _, fc_req, off_req = calibrated_forecast(req, cal_start, split, horizon)

    # ---- RL agent ------------------------------------------------------------
    # RL may pick any buffer that is still safe for the latency SLA
    targets = allowed_targets(max_util_for_latency(max_rt))
    rl = QLearningScaler(cpu_cal_fc, mem_cal_fc, pod_mem, targets, max_rt, sla_weight=sla_weight)
    rl_hist = rl.train(cpu, cal_start, split, env_kwargs, episodes=rl_episodes,
                       progress=lambda e, E: say(f"Training RL agent: episode {e}/{E}", 0.22 + 0.58 * e / E))

    # ---- final forecasts for the test week ----------------------------------
    say("Forecasting the unseen test week ...", 0.82)
    forecasts = {}
    for key, fc, off in [("cpu", fc_cpu, off_cpu), ("memory", fc_mem, off_mem), ("requests", fc_req, off_req)]:
        med, up = fc.pred, fc.pred + off
        forecasts[key] = {"median": med, "upper": up, "actual": fc.y_all, "model": fc,
                          "metrics": fc.evaluate(split, n - horizon, up), "offset": off}
    cpu_fc, mem_fc = forecasts["cpu"]["upper"], forecasts["memory"]["upper"]
    rl.cpu_fc, rl.mem_fc = cpu_fc, mem_fc

    # ---- anomaly detection --------------------------------------------------
    say("Running anomaly detection ...", 0.86)
    anomalies, anomaly_points, anomaly_threshold = detect_anomalies(cpu, req, mem, 0, split, split, n)
    anomaly_eval = evaluate_against_labels(anomalies, wl["injected"].to_numpy(), split, n)

    # ---- compare autoscalers ------------------------------------------------
    say("Simulating all autoscalers on the test week ...", 0.9)
    peak_pods = max(math.ceil(cpu[:split].max() * 1.10 / pod_cpu),
                    math.ceil(mem[:split].max() / (pod_mem * MEM_TARGET)))
    policies = [StaticPolicy(peak_pods), ReactivePolicy(high=max_util),
                PredictivePolicy(cpu_fc, mem_fc, pod_mem, target_util), rl]
    days = test_len / STEPS_PER_DAY
    price_step_vcpu = price_hr * STEP_MIN / 60 / pod_cpu
    runs, rows = {}, []
    for p in policies:
        df = enrich(simulate(p, cpu, split, n, env_kwargs), mem, pod_mem, max_rt)
        runs[p.name] = df
        rows.append(summarize(df, p.name, days, price_step_vcpu, sla_weight, pod_cpu))
    summary = pd.DataFrame(rows).set_index("Policy")

    cfg = dict(pod_cpu=pod_cpu, pod_mem=pod_mem, price_hr=price_hr, boot_steps=boot_steps, max_rt=max_rt,
               avail_target=avail_target, max_util=max_util, target_util=target_util, sla_weight=sla_weight)
    R = {
        "wl": wl, "cpu": cpu, "mem": mem, "req": req, "n": n, "split": split, "cal_start": cal_start,
        "horizon": horizon, "days": days, "cfg": cfg, "env_kwargs": env_kwargs,
        "forecasts": forecasts, "rl": rl, "rl_history": rl_hist, "targets": targets,
        "anomalies": anomalies, "anomaly_points": anomaly_points, "anomaly_eval": anomaly_eval,
        "anomaly_threshold": anomaly_threshold,
        "runs": runs, "summary": summary, "price_step_vcpu": price_step_vcpu,
    }
    R["decisions"] = explain_decisions(R)
    say("Done", 1.0)
    return R


# --------------------------------------------------------------------------
# Explainable AI: a human-readable reason for every scaling decision
# --------------------------------------------------------------------------
def explain_decisions(R):
    cfg, rl = R["cfg"], R["rl"]
    df = R["runs"][HYBRID]
    cpu_fc, mem_fc = R["forecasts"]["cpu"]["upper"], R["forecasts"]["memory"]["upper"]
    req = R["req"]
    flagged = set(R["anomaly_points"].loc[R["anomaly_points"]["flag"], "t"])
    price_step = cfg["price_hr"] * STEP_MIN / 60
    out = []
    for i, lg in enumerate(rl.log):
        t = lg["t"]
        before = lg["pods_before"]
        after = int(df["pods"].iloc[i])
        if after == before:
            continue
        need, by_cpu, by_mem = pods_needed(cpu_fc[t], mem_fc[t], cfg["pod_cpu"], cfg["pod_mem"], lg["target"])
        pred_util_now = cpu_fc[t] / max(before * cfg["pod_cpu"], 1)
        pred_rt_now = float(response_time(pred_util_now))
        pred_util_new = cpu_fc[t] / (after * cfg["pod_cpu"])
        req_change = (req[t - 1] - req[t - 4]) / max(req[t - 4], 1)
        up = after > before
        reasons = [f"Predicted peak CPU (next {R['horizon'] * STEP_MIN} min): {cpu_fc[t]:.0f} vCPU = "
                   f"{100 * pred_util_now:.0f}% of current capacity"]
        if abs(req_change) > 0.03:
            reasons.append(f"Request rate {'increasing' if req_change > 0 else 'decreasing'} "
                           f"({100 * req_change:+.0f}% in last 15 min, now {req[t - 1]:,.0f} req/s)")
        if up and pred_rt_now > cfg["max_rt"]:
            reasons.append(f"SLA at risk: predicted response time {pred_rt_now:,.0f} ms > {cfg['max_rt']} ms limit")
        if up and by_mem > by_cpu:
            reasons.append(f"Memory forecast {mem_fc[t]:.0f} GiB needs {by_mem} pods (memory-bound)")
        if (t - 1) in flagged:
            reasons.append("Anomaly detector flagged unusual behaviour - keeping extra headroom")
        if lg["missed"]:
            reasons.append("Capacity was saturated in the previous interval")
        reasons.append(f"RL agent chose {100 * lg['target']:.0f}% target utilization "
                       f"({VOL_NAMES[lg['vol']]} traffic, {TREND_NAMES[lg['trend']]} trend, "
                       f"{HOUR_NAMES[lg['hour_b']]} h)")
        if not up:
            reasons.append(f"Releasing {before - after} pod(s) saves ${(before - after) * price_step * 12:.2f}/hour "
                           f"while keeping utilization at {100 * pred_util_new:.0f}%")
        else:
            reasons.append(f"Adding {after - before} pod(s) brings predicted utilization to {100 * pred_util_new:.0f}%")
        out.append({"t": t, "action": "Scale Up" if up else "Scale Down", "from": before, "to": after,
                    "pred_cpu": cpu_fc[t], "pred_mem": mem_fc[t], "pred_util_now": pred_util_now,
                    "pred_util_new": pred_util_new, "pred_rt_now": pred_rt_now, "target": lg["target"],
                    "by_cpu": by_cpu, "by_mem": by_mem, "reasons": reasons})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# What-if simulation (does not touch the "live" results)
# --------------------------------------------------------------------------
class ReplayPolicy:
    name = "Keep current plan"

    def __init__(self, pods, start):
        self.pods, self.start = pods, start

    def reset(self):
        pass

    def decide(self, t, cpu, env):
        return int(self.pods[t - self.start])


def what_if(R, traffic_pct=30, pod_cpu=None):
    cfg = R["cfg"]
    mult = 1 + traffic_pct / 100
    pod_cpu = pod_cpu or cfg["pod_cpu"]
    split, n = R["split"], R["n"]
    cpu = R["cpu"] * mult
    mem = R["mem"] * (1 + 0.6 * (mult - 1))     # memory grows slower than traffic (fixed overheads)
    env_kwargs = dict(R["env_kwargs"], pod_cpu=pod_cpu)
    price_step_vcpu = cfg["price_hr"] * STEP_MIN / 60 / pod_cpu

    rl = copy.copy(R["rl"])
    rl.cpu_fc = R["forecasts"]["cpu"]["upper"] * mult
    rl.mem_fc = R["forecasts"]["memory"]["upper"] * (1 + 0.6 * (mult - 1))
    base = R["runs"][HYBRID]
    res = {}
    for name, pol in [("AI plan for new traffic", rl),
                      ("Keep current plan", ReplayPolicy(base["pods"].to_numpy(), split))]:
        df = enrich(simulate(pol, cpu, split, n, env_kwargs), mem, cfg["pod_mem"], cfg["max_rt"])
        res[name] = (df, summarize(df, name, R["days"], price_step_vcpu, cfg["sla_weight"], pod_cpu))
    return res
