"""
AI components:

1. Forecaster - Gradient-Boosted trees that predict the PEAK of a metric
   (CPU, memory, requests) over the next few minutes (the pod start-up
   window), plus a conformal upper bound so we provision for uncertainty.

2. PredictivePolicy - scale BEFORE the spike: replicas = forecast / (pod size x
   target utilization), checking both CPU and memory.

3. QLearningScaler - a Reinforcement-Learning agent that learns WHICH safety
   buffer to keep in each situation, rewarded for low cost and punished for
   SLA breaches (dropped requests, slow responses).
"""
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .simulator import Cluster, STEPS_PER_DAY, STEP_MIN, response_time

LAGS = [1, 2, 3, 6, 12, 24, 36, STEPS_PER_DAY, 7 * STEPS_PER_DAY]
MEM_TARGET = 0.90  # never plan memory above 90% of pod limits


# --------------------------------------------------------------------------
# 1. Forecaster
# --------------------------------------------------------------------------
FEATURE_LABELS = {
    "lag_1": "Value 5 min ago", "lag_2": "Value 10 min ago", "lag_3": "Value 15 min ago",
    "lag_6": "Value 30 min ago", "lag_12": "Value 1 h ago", "lag_24": "Value 2 h ago",
    "lag_36": "Value 3 h ago", f"lag_{STEPS_PER_DAY}": "Same time yesterday",
    f"lag_{7 * STEPS_PER_DAY}": "Same time last week", "mean_30min": "Average last 30 min",
    "mean_1h": "Average last 1 h", "std_1h": "Volatility last 1 h", "slope_15min": "Trend last 15 min",
    "yesterday_next_15min": "Yesterday, 15 min later", "lastweek_next_15min": "Last week, 15 min later",
    "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)", "day_of_week": "Day of week",
}


def make_features(series):
    """Features for decision time t use ONLY values before t (no peeking ahead)."""
    s = pd.Series(series)
    n = len(s)
    t = np.arange(n)
    past = s.shift(1)
    f = {}
    for k in LAGS:
        f[f"lag_{k}"] = s.shift(k)
    f["mean_30min"] = past.rolling(6).mean()
    f["mean_1h"] = past.rolling(12).mean()
    f["std_1h"] = past.rolling(12).std()
    f["slope_15min"] = s.shift(1) - s.shift(4)
    f["yesterday_next_15min"] = s.shift(STEPS_PER_DAY - 3)
    f["lastweek_next_15min"] = s.shift(7 * STEPS_PER_DAY - 3)
    hour = (t % STEPS_PER_DAY) * STEP_MIN / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["day_of_week"] = (t // STEPS_PER_DAY) % 7
    return pd.DataFrame(f)


def make_target(series, horizon):
    """Peak value over [t, t+horizon): what capacity decided at t must cover."""
    s = pd.Series(series)
    return s[::-1].rolling(horizon, min_periods=1).max()[::-1].values


class Forecaster:
    """Gradient-boosted trees point forecast of the peak over the next
    `horizon` steps. The uncertainty band is added afterwards by conformal
    calibration (see `calibrated_forecast`)."""

    def __init__(self, horizon=3):
        self.horizon = horizon
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   l2_regularization=1.0, random_state=0)

    def fit(self, series, train_end):
        X = make_features(series)
        y = make_target(series, self.horizon)
        lo, hi = 12, train_end - self.horizon
        self.model.fit(X.iloc[lo:hi], y[lo:hi])
        self.X_all, self.y_all = X, y
        self.pred = self.model.predict(X)  # each row only uses past data
        return self

    def evaluate(self, start, end, upper):
        y = self.y_all[start:end]
        m = self.pred[start:end]
        naive = self.X_all["lag_1"].values[start:end]
        return {
            "MAE": float(np.mean(np.abs(y - m))),
            "MAPE (%)": float(100 * np.mean(np.abs(y - m) / y)),
            "Naive MAPE (%)": float(100 * np.mean(np.abs(y - naive) / y)),
            "Coverage (%)": float(100 * np.mean(y <= upper[start:end])),
        }

    def feature_importance(self, start, end, n=1500):
        idx = np.linspace(start, end - self.horizon - 1, n).astype(int)
        r = permutation_importance(self.model, self.X_all.iloc[idx], self.y_all[idx],
                                   n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
        return (pd.DataFrame({"feature": [FEATURE_LABELS.get(c, c) for c in self.X_all.columns],
                              "importance": r.importances_mean})
                .sort_values("importance", ascending=True))


def calibrated_forecast(series, cal_start, split, horizon, coverage=0.95):
    """Conformal prediction: fit on data before `cal_start`, measure the
    forecast errors on the held-out [cal_start, split) week and take their
    `coverage` quantile as a safety margin, so the upper bound covers the true
    peak ~95% of the time. Returns (cal-fold upper bound, final model, margin)."""
    fc_cal = Forecaster(horizon).fit(series, cal_start)
    resid = fc_cal.y_all[cal_start:split - horizon] - fc_cal.pred[cal_start:split - horizon]
    margin = max(0.0, float(np.quantile(resid, coverage)))
    fc = Forecaster(horizon).fit(series, split)
    return fc_cal.pred + margin, fc, margin


# --------------------------------------------------------------------------
# 2. Predictive autoscaler
# --------------------------------------------------------------------------
def pods_needed(cpu_fc, mem_fc, pod_cpu, pod_mem, target_util):
    by_cpu = math.ceil(cpu_fc / (pod_cpu * target_util))
    by_mem = math.ceil(mem_fc / (pod_mem * MEM_TARGET))
    return max(by_cpu, by_mem), by_cpu, by_mem


class Stabilizer:
    """Kubernetes-HPA-style damping: scale up immediately, but scale down only
    (a) to the HIGHEST recommendation of the last few steps (stabilization
    window) and (b) when the change is more than `tolerance` (HPA default 10%).
    Prevents pods flapping up and down every 5 minutes."""

    def __init__(self, window=3, tolerance=0.10):
        self.window, self.tolerance = window, tolerance
        self.recent = []

    def __call__(self, desired, current):
        self.recent = (self.recent + [desired])[-self.window:]
        if desired >= current:
            return desired
        target = max(self.recent)
        return target if target < current * (1 - self.tolerance) else current


class PredictivePolicy:
    """Provision for the calibrated forecast peak at a fixed target utilization."""
    name = "AI Predictive (ML forecast)"

    def __init__(self, cpu_fc, mem_fc, pod_mem, target_util):
        self.cpu_fc, self.mem_fc, self.pod_mem, self.target_util = cpu_fc, mem_fc, pod_mem, target_util

    def reset(self):
        self.stab = Stabilizer()

    def decide(self, t, cpu, env):
        need = pods_needed(self.cpu_fc[t], self.mem_fc[t], env.pod_cpu, self.pod_mem, self.target_util)[0]
        return self.stab(need, env.total)


# --------------------------------------------------------------------------
# 3. Hybrid Reinforcement-Learning autoscaler (tabular Q-learning)
# --------------------------------------------------------------------------
# A fixed rule always keeps the same safety buffer. The RL agent LEARNS which
# buffer (target utilization) to use in each situation:
#   state  = time of day, demand trend, demand volatility, "did we just breach?"
#   action = target CPU utilization applied to the ML forecast
#   reward = -(pod cost + SLA penalty for dropped requests / slow responses)
ALL_TARGETS = [0.55, 0.62, 0.68, 0.74, 0.80, 0.85, 0.90, 0.95]
TREND_NAMES = ["falling", "flat", "rising"]
VOL_NAMES = ["calm", "normal", "volatile"]
HOUR_NAMES = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]


def allowed_targets(max_util):
    t = [x for x in ALL_TARGETS if x <= max_util + 1e-9]
    return t[-6:] if len(t) >= 3 else ALL_TARGETS[:3]


class QLearningScaler:
    name = "AI Hybrid (Forecast + RL)"

    def __init__(self, cpu_fc, mem_fc, pod_mem, targets, max_rt, sla_weight=10.0,
                 alpha=0.1, gamma=0.5, seed=0):
        self.cpu_fc, self.mem_fc, self.pod_mem = cpu_fc, mem_fc, pod_mem
        self.targets = targets
        self.max_rt = max_rt
        self.max_load = 1 - 60 / max_rt  # utilization above which latency breaks the SLA
        self.n_states = 6 * 3 * 3 * 2
        self.Q = np.zeros((self.n_states, len(targets)))
        self.visits = np.zeros(self.n_states, dtype=int)
        self.sla_weight = sla_weight
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.history = []

    # --- state encoding ---------------------------------------------------
    def _precompute(self, cpu):
        """Vectorised per-step features (hour, trend, volatility) - computed once."""
        if getattr(self, "_src", None) is cpu:
            return
        s = pd.Series(cpu)
        t = np.arange(len(cpu))
        prev = s.shift(4).clip(lower=1)
        self._change = ((s.shift(1) - prev) / prev).fillna(0).to_numpy()
        trend = np.where(self._change < -0.05, 0, np.where(self._change > 0.05, 2, 1))
        past = s.shift(1)
        cv = (past.rolling(12, min_periods=1).std(ddof=0) / past.rolling(12, min_periods=1).mean().clip(lower=1))
        cv = cv.fillna(0).to_numpy()
        vol = np.where(cv < 0.05, 0, np.where(cv < 0.12, 1, 2))
        hour_b = ((t % STEPS_PER_DAY) * STEP_MIN // 60) // 4
        self._feat = np.stack([hour_b, trend, vol], axis=1).astype(int)
        self._base_state = ((hour_b * 3 + trend) * 3 + vol) * 2
        self._src = cpu

    def features(self, t, cpu, capacity):
        self._precompute(cpu)
        h, tr, v = self._feat[t]
        missed = 1 if cpu[t - 1] > 0.98 * capacity else 0
        return h, tr, v, missed, self._change[t]

    def state(self, t, cpu, env):
        self._precompute(cpu)
        return int(self._base_state[t]) + (1 if cpu[t - 1] > 0.98 * env.capacity else 0)

    def target_pods(self, t, a, env):
        return pods_needed(self.cpu_fc[t], self.mem_fc[t], env.pod_cpu, self.pod_mem, self.targets[a])[0]

    def reward(self, rec, env):
        # cost in "pod-steps"; dropped vCPU and slow responses are penalised
        slow = 1.0 if rec["load"] > self.max_load else 0.0
        return -(env.total + self.sla_weight * (rec["unserved"] / env.pod_cpu + 0.25 * slow))

    # --- training ------------------------------------------------------------
    def train(self, cpu, start, end, env_kwargs, episodes=40, progress=None):
        for ep in range(episodes):
            eps = max(0.02, 1.0 - ep / (0.7 * episodes))
            env = Cluster(**env_kwargs)
            env.reset(math.ceil(cpu[start] / (env.pod_cpu * 0.7)))
            stab = Stabilizer()
            s = self.state(start, cpu, env)
            total_r, cost, unserved = 0.0, 0.0, 0.0
            for t in range(start, end - 1):
                a = self.rng.integers(len(self.targets)) if self.rng.random() < eps else int(np.argmax(self.Q[s]))
                rec = env.step(t, stab(self.target_pods(t, a, env), env.total), cpu[t])
                r = self.reward(rec, env)
                s2 = self.state(t + 1, cpu, env)
                self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2].max() - self.Q[s, a])
                self.visits[s] += 1
                s = s2
                total_r += r
                cost += rec["cost"]
                unserved += rec["unserved"]
            self.history.append({"episode": ep + 1, "reward": total_r, "epsilon": eps, "cost": cost,
                                 "unserved_pct": 100 * unserved / cpu[start:end].sum()})
            if progress:
                progress(ep + 1, episodes)
        return pd.DataFrame(self.history)

    # --- acting (greedy) -------------------------------------------------------
    def reset(self):
        self.log = []
        self.stab = Stabilizer()

    def decide(self, t, cpu, env):
        s = self.state(t, cpu, env)
        a = int(np.argmax(self.Q[s]))
        h, tr, v, m, ch = self.features(t, cpu, env.capacity)
        self.log.append({"t": t, "target": self.targets[a], "hour_b": h, "trend": tr, "vol": v,
                         "missed": m, "change": ch, "pods_before": env.total})
        return self.stab(self.target_pods(t, a, env), env.total)

    def policy_table(self):
        """Learned target utilization per (hour, volatility) with no recent
        breach, visit-weighted over trend - for the dashboard heat-map."""
        out = np.full((3, 6), np.nan)
        for h in range(6):
            for v in range(3):
                qs, w = [], []
                for tr in range(3):
                    s = ((h * 3 + tr) * 3 + v) * 2 + 0
                    if self.visits[s]:
                        qs.append(self.Q[s]); w.append(self.visits[s])
                if qs:
                    out[v, h] = 100 * self.targets[int(np.argmax(np.average(qs, axis=0, weights=w)))]
        return pd.DataFrame(out, index=VOL_NAMES, columns=HOUR_NAMES)
