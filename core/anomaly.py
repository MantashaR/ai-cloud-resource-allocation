"""
AI anomaly detection (unsupervised, multivariate).

Signals, each compared with the same time one week earlier:
  * sudden traffic jump    - request rate vs. the previous hour
  * traffic deviation      - request rate vs. last week
  * memory deviation       - memory vs. last week, after removing the part that
                             is explained by traffic (memory normally follows load)
  * CPU efficiency         - CPU per request (CPU rising while requests do not
                             = runaway process / abnormal CPU)

A Robust Covariance model (Minimum Covariance Determinant, scikit-learn
EllipticEnvelope) learns what "normal" looks like from history, ignoring the
outliers in it. A point's Mahalanobis distance from that normal cloud is its
anomaly score; consecutive anomalous points are grouped into events with a
type, severity and timestamp.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope

from .simulator import STEPS_PER_DAY, REQ_PER_VCPU

FEATURES = {
    "jump": "Traffic spike",
    "traffic": "Traffic spike",
    "memory": "Memory spike",
    "cpu_eff": "Abnormal CPU",
}


def _weekly_dev(x):
    """Relative deviation from the same 15-minute window one week earlier
    (same weekday, so weekends are compared with weekends)."""
    s = pd.Series(x)
    week = 7 * STEPS_PER_DAY
    expected = s.shift(week).rolling(3, min_periods=1).mean()
    smooth = s.rolling(3, min_periods=1).mean()  # 15-min smoothing removes single-point jitter
    floor = 0.5 * float(np.nanmedian(x))         # avoid huge % swings at night when values are tiny
    return ((smooth - expected) / expected.clip(lower=floor)).values


def _weekly_abs_dev(x):
    s = pd.Series(x)
    week = 7 * STEPS_PER_DAY
    return (s.rolling(3, min_periods=1).mean() - s.shift(week).rolling(3, min_periods=1).mean()).values


def detect_anomalies(cpu, requests, memory, ref_start, ref_end, start, end):
    traffic = _weekly_dev(requests)
    cpu_eff = pd.Series(cpu / (requests / REQ_PER_VCPU)).rolling(2, min_periods=1).mean().values - 1
    ref = np.arange(max(ref_start, 7 * STEPS_PER_DAY + 2), ref_end)

    # memory normally follows traffic: extra GiB per extra request/s is learned
    # from history and only the unexplained part of the memory change is kept
    mem_d, req_d = _weekly_abs_dev(memory), _weekly_abs_dev(requests)
    beta = np.polyfit(req_d[ref], mem_d[ref], 1)
    mem_resid = mem_d - np.polyval(beta, req_d)

    r = pd.Series(requests)
    jump = (r.rolling(2, min_periods=1).mean() / r.shift(3).rolling(12, min_periods=1).mean() - 1).values

    raw = pd.DataFrame({"jump": jump, "traffic": traffic, "memory": mem_resid, "cpu_eff": cpu_eff})
    med = raw.iloc[ref].median()
    mad = (raw.iloc[ref] - med).abs().median() * 1.4826 + 1e-9
    z = (raw - med) / mad

    model = EllipticEnvelope(support_fraction=0.9, random_state=0).fit(z.iloc[ref])
    dist_ref = np.sqrt(model.mahalanobis(z.iloc[ref]))
    threshold = max(5.0, float(np.quantile(dist_ref, 0.99)))

    zt = z.iloc[start:end].fillna(0)
    dist = np.sqrt(model.mahalanobis(zt))
    points = pd.DataFrame({"t": np.arange(start, end), "score": dist, "flag": dist > threshold})
    points = pd.concat([points, zt.reset_index(drop=True)], axis=1)

    # group consecutive flagged points (gaps <= 2 steps) into events
    events, cur = [], None
    for _, r in points[points["flag"]].iterrows():
        if cur and r["t"] - cur["end"] <= 3:
            cur["end"] = r["t"]
            cur["rows"].append(r)
        else:
            if cur:
                events.append(cur)
            cur = {"start": r["t"], "end": r["t"], "rows": [r]}
    if cur:
        events.append(cur)

    out = []
    for e in events:
        rows = pd.DataFrame(e["rows"])
        zabs = rows[list(FEATURES)].abs().max()
        top = zabs.idxmax()
        kind = FEATURES[top]
        sign = np.sign(rows.loc[rows[top].abs().idxmax(), top])
        if sign < 0:
            if top not in ("jump", "traffic"):
                continue  # resources dropping below normal is harmless
            kind = "Traffic drop"
        peak = float(rows["score"].max())
        if len(rows) == 1 and peak < 1.5 * threshold:
            continue  # ignore one-off blips
        sev = "High" if peak >= 2.5 * threshold else ("Medium" if peak >= 1.5 * threshold else "Low")
        out.append({"start_t": int(e["start"]), "end_t": int(e["end"]), "type": kind, "severity": sev,
                    "score": peak, "duration_min": 5 * (int(e["end"]) - int(e["start"]) + 1),
                    "jump_z": float(rows["jump"].iloc[rows["score"].argmax()]),
                    "traffic_z": float(rows["traffic"].iloc[rows["score"].argmax()]),
                    "memory_z": float(rows["memory"].iloc[rows["score"].argmax()]),
                    "cpu_eff_z": float(rows["cpu_eff"].iloc[rows["score"].argmax()])})
    cols = ["start_t", "end_t", "type", "severity", "score", "duration_min", "jump_z", "traffic_z", "memory_z", "cpu_eff_z"]
    return pd.DataFrame(out, columns=cols), points, threshold


def evaluate_against_labels(events, labels, start, end):
    """Event-level check against the anomalies injected into the test data."""
    lab = pd.Series(labels[start:end], index=np.arange(start, end))
    truth, cur = [], None
    for t, v in lab.items():
        if v and (cur is None or cur["type"] != v or t != cur["end"] + 1):
            if cur:
                truth.append(cur)
            cur = {"type": v, "start": t, "end": t}
        elif v:
            cur["end"] = t
    if cur:
        truth.append(cur)
    rows = []
    for g in truth:
        hit = events[(events["start_t"] <= g["end"] + 2) & (events["end_t"] >= g["start"] - 2)]
        rows.append({"Injected anomaly": g["type"], "start_t": g["start"],
                     "Detected": "Yes" if len(hit) else "No",
                     "Detected as": ", ".join(sorted(set(hit["type"]))) if len(hit) else "-",
                     "Delay (min)": 5 * max(0, int(hit["start_t"].min()) - g["start"]) if len(hit) else None})
    return pd.DataFrame(rows)
