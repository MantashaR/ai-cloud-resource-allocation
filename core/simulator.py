"""
Kubernetes-style cluster simulator + workload generator + baseline autoscalers.

Time is discrete: 1 step = 5 minutes (288 steps per day).
The application is served by identical pods (replicas) of `pod_cpu` vCPU and
`pod_mem` GiB. A newly requested pod needs `boot_steps` steps before it can
serve traffic (node provisioning + image pull + start-up) and is billed while
it starts. This delay is what makes *prediction* valuable: a purely reactive
autoscaler (like the default Kubernetes HPA) is always late.
"""
import math
import numpy as np
import pandas as pd

STEP_MIN = 5
STEPS_PER_DAY = 24 * 60 // STEP_MIN  # 288
RT_BASE_MS = 60       # response time of an idle pod
RT_OVERLOAD_MS = 3000  # response time when pods are saturated (requests time out)
REQ_PER_VCPU = 25      # requests/second one vCPU can handle


def response_time(load):
    """Queueing-theory style latency: grows sharply as utilization -> 100%."""
    load = np.asarray(load, dtype=float)
    rt = RT_BASE_MS / np.clip(1 - load, 0.02, None)
    return np.where(load >= 0.98, RT_OVERLOAD_MS, np.minimum(rt, RT_OVERLOAD_MS))


def max_util_for_latency(max_rt_ms):
    return 1 - RT_BASE_MS / max_rt_ms


# --------------------------------------------------------------------------
# Workload generation
# --------------------------------------------------------------------------
SCENARIOS = {
    "E-commerce (daily + weekly pattern)": "normal",
    "Flash sale (sudden unseen surge)": "flash",
    "Steady growth (startup going viral)": "growth",
}


def generate_workload(days=28, seed=42, scenario="normal"):
    """Synthetic but realistic cluster metrics at 5-minute resolution:
    cpu (vCPU demanded), requests (req/s), memory (GiB), plus a ground-truth
    label of the anomalies injected into the final (test) week."""
    rng = np.random.default_rng(seed)
    n = days * STEPS_PER_DAY
    t = np.arange(n)
    hour = (t % STEPS_PER_DAY) * STEP_MIN / 60
    dow = (t // STEPS_PER_DAY) % 7

    # Office-hours peak around 14:00 plus an evening peak around 20:30
    daily = 0.55 * np.exp(-((hour - 14) / 3.5) ** 2) + 0.30 * np.exp(-((hour - 20.5) / 2) ** 2)
    weekend = np.where(dow >= 5, 0.65, 1.0)
    growth_rate = 0.60 if scenario == "growth" else 0.15
    growth = 1 + growth_rate * t / n
    traffic = (25 + 110 * daily * weekend) * growth

    # Correlated noise (real traffic is "sticky", not white noise)
    noise = np.zeros(n)
    eps = rng.normal(0, 3, n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + eps[i]
    traffic = traffic + noise

    # Random short bursts (marketing pushes, retries, bots ...)
    for _ in range(days * 2):
        start = rng.integers(0, n - 14)
        dur = int(rng.integers(4, 14))
        traffic[start:start + dur] += rng.uniform(15, 45) * np.hanning(dur)

    label = np.array([""] * n, dtype=object)

    def at(day, hr):
        return day * STEPS_PER_DAY + int(hr * 60 // STEP_MIN)

    # --- anomalies injected into the final week (ground truth for evaluation) ---
    s = at(days - 4, 15)                          # sudden viral traffic spike (~2.5x), 40 min
    traffic[s:s + 8] += 1.6 * traffic[s:s + 8] * np.hanning(8).clip(0.5)
    label[s:s + 8] = "Traffic spike"
    if scenario == "flash":
        for day, hr in [(days - 5, 18), (days - 2, 11)]:   # 3-hour flash sales
            s = at(day, hr)
            dur = 36
            ramp = np.concatenate([np.linspace(0, 1, 6), np.ones(dur - 12), np.linspace(1, 0, 6)])
            traffic[s:s + dur] += 0.9 * traffic[s:s + dur] * ramp
            label[s:s + dur] = "Traffic spike"

    traffic = np.clip(traffic, 2, None)
    requests = traffic * REQ_PER_VCPU * (1 + rng.normal(0, 0.02, n))
    cpu = traffic.copy()

    s = at(days - 6, 10)                          # runaway process: CPU up, requests flat
    cpu[s:s + 9] += 35
    label[s:s + 9] = "Abnormal CPU"

    # Memory: per-request working set + a cache that grows all day and is
    # flushed at 03:00, plus one memory leak that is fixed by a restart.
    cache = 14 * (((t - at(0, 3)) % STEPS_PER_DAY) / STEPS_PER_DAY)
    memory = 10 + 0.45 * traffic + cache + rng.normal(0, 0.6, n)
    s = at(days - 3, 1)
    leak = np.linspace(0, 60, 36)
    memory[s:s + 36] += leak
    label[s:s + 36] = "Memory spike"

    times = pd.date_range("2026-01-05", periods=n, freq=f"{STEP_MIN}min")  # a Monday
    return pd.DataFrame({"time": times, "cpu": cpu.round(2), "requests": requests.round(1),
                         "memory": memory.round(2), "injected": label})


# --------------------------------------------------------------------------
# Cluster environment
# --------------------------------------------------------------------------
class Cluster:
    def __init__(self, pod_cpu=4, price_hr=0.20, boot_steps=2, min_pods=1, max_pods=150):
        self.pod_cpu = pod_cpu
        self.price_step = price_hr * STEP_MIN / 60
        self.boot_steps = boot_steps
        self.min_pods = min_pods
        self.max_pods = max_pods
        self.reset(min_pods)

    def reset(self, init_pods):
        self.running = int(np.clip(init_pods, self.min_pods, self.max_pods))
        self.pending = []  # ready-times of pods that are still starting

    @property
    def total(self):
        return self.running + len(self.pending)

    @property
    def capacity(self):
        return self.running * self.pod_cpu

    def step(self, t, desired, demand):
        """Apply the scaling decision, advance one step, serve `demand` vCPU."""
        desired = int(np.clip(desired, self.min_pods, self.max_pods))
        if desired > self.total:
            self.pending += [t + self.boot_steps] * (desired - self.total)
        elif desired < self.total:
            remove = self.total - desired
            while remove and self.pending:          # cancel newest starts first
                self.pending.pop()
                remove -= 1
            self.running -= remove                  # then terminate running pods

        ready = sum(1 for r in self.pending if r <= t)
        if ready:
            self.pending = [r for r in self.pending if r > t]
            self.running += ready

        cap = self.capacity
        served = min(demand, cap)
        return {
            "pods_running": self.running,
            "pods_starting": len(self.pending),
            "capacity": cap,
            "served": served,
            "unserved": demand - served,
            "load": demand / cap if cap else 9.9,
            "cost": (self.running + len(self.pending)) * self.price_step,
        }


def simulate(policy, cpu, start, end, env_kwargs, init_pods=None):
    """Run a policy over cpu[start:end]. Policies may only look at data before t."""
    env = Cluster(**env_kwargs)
    if init_pods is None:
        init_pods = math.ceil(cpu[start] / (env.pod_cpu * 0.7))
    env.reset(init_pods)
    policy.reset()
    rows = []
    for t in range(start, end):
        desired = policy.decide(t, cpu, env)
        rec = env.step(t, desired, cpu[t])
        rec["t"] = t
        rec["cpu"] = cpu[t]
        rows.append(rec)
    return pd.DataFrame(rows)


def enrich(df, memory, pod_mem, max_rt):
    """Add memory, latency and SLA columns to a simulation result."""
    df = df.copy()
    df["pods"] = df["pods_running"] + df["pods_starting"]
    df["memory"] = memory[df["t"].values]
    df["mem_capacity"] = df["pods_running"] * pod_mem
    df["mem_util"] = (df["memory"] / df["mem_capacity"]).clip(upper=9.9)
    df["oom"] = df["memory"] > df["mem_capacity"]
    df["cpu_util"] = df["load"].clip(upper=1.0)
    df["response_ms"] = response_time(df["load"])
    df["sla_breach"] = (df["unserved"] > 0.01) | (df["response_ms"] > max_rt) | df["oom"]
    return df


def summarize(df, name, days, price_step_vcpu, sla_weight, pod_cpu):
    served_frac = 1 - df["unserved"].sum() / df["cpu"].sum()
    breach_pen = sla_weight * price_step_vcpu * (
        df["unserved"].sum() + 0.25 * pod_cpu * (df["sla_breach"] & (df["unserved"] <= 0.01)).sum())
    cost = df["cost"].sum()
    return {
        "Policy": name,
        "Cost ($)": cost,
        "Monthly cost ($)": cost / days * 30,
        "SLA penalty ($)": breach_pen,
        "Cost + penalty ($)": cost + breach_pen,
        "SLA breaches (%)": 100 * df["sla_breach"].mean(),
        "Availability (%)": 100 * served_frac * (1 - df["oom"].mean()),
        "p95 response (ms)": float(np.percentile(df["response_ms"], 95)),
        "Avg CPU util (%)": 100 * df["cpu_util"].mean(),
        "Avg memory util (%)": 100 * df["mem_util"].clip(upper=1).mean(),
        "Avg pods": df["pods"].mean(),
        "Peak pods": int(df["pods"].max()),
        "Scaling events": int((df["pods"].diff().fillna(0) != 0).sum()),
    }


# --------------------------------------------------------------------------
# Baseline policies (what clusters do today without AI)
# --------------------------------------------------------------------------
class StaticPolicy:
    """Over-provision for the historical peak - safe but wasteful."""
    name = "Static (peak provisioning)"

    def __init__(self, pods):
        self.pods = pods

    def reset(self):
        pass

    def decide(self, t, cpu, env):
        return self.pods


class ReactivePolicy:
    """Classic threshold autoscaler (default Kubernetes HPA / AWS step scaling):
    +2 pods if CPU > high, -1 pod if CPU < low, with a cooldown."""
    name = "Reactive (HPA threshold)"

    def __init__(self, high=0.80, low=0.40, up=2, down=1, cooldown=3):
        self.high, self.low, self.up, self.down, self.cooldown = high, low, up, down, cooldown

    def reset(self):
        self.last_action = -10**9

    def decide(self, t, cpu, env):
        if t - self.last_action < self.cooldown or env.capacity == 0:
            return env.total
        load = cpu[t - 1] / env.capacity
        if load > self.high:
            self.last_action = t
            return env.total + self.up
        if load < self.low:
            self.last_action = t
            return env.total - self.down
        return env.total
