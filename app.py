"""
AI-Driven Intelligent Cloud Resource Allocation & Predictive Autoscaling
Run:  streamlit run app.py
"""
import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from core.simulator import generate_workload, SCENARIOS, STEPS_PER_DAY, STEP_MIN, REQ_PER_VCPU, max_util_for_latency
from core.pipeline import run_experiment, what_if, HYBRID

st.set_page_config(page_title="AI Cloud Resource Allocation", page_icon="☁️", layout="wide")

STATIC, REACTIVE, PREDICT = "Static (peak provisioning)", "Reactive (HPA threshold)", "AI Predictive (ML forecast)"
ORDER = [STATIC, REACTIVE, PREDICT, HYBRID]
COLORS = {STATIC: "#9CA3AF", REACTIVE: "#F59E0B", PREDICT: "#3B82F6", HYBRID: "#10B981"}
SHORT = {STATIC: "Static", REACTIVE: "Reactive HPA", PREDICT: "AI Forecast", HYBRID: "AI Forecast+RL"}
SEV_COLORS = {"High": "#EF4444", "Medium": "#F97316", "Low": "#EAB308"}
INK = "#111827"
PLOT = dict(margin=dict(t=50, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.18))

st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
.hero {background: linear-gradient(120deg,#4F46E5 0%,#0EA5E9 60%,#10B981 100%);
       color:white;padding:20px 28px;border-radius:16px;margin-bottom:14px;}
.hero h1 {color:white;margin:0;font-size:1.9rem;}
.hero p {margin:6px 0 0 0;opacity:.93;font-size:1rem;}
.kpi {border-radius:14px;padding:14px 18px;border:1px solid rgba(128,128,128,.25);height:100%;}
.kpi .v {font-size:1.8rem;font-weight:700;line-height:1.2}
.kpi .l {font-size:.85rem;opacity:.75;}
.card {border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:16px 20px;}
.vmgrid {display:flex;flex-wrap:wrap;gap:4px;min-height:44px;}
.vm {width:16px;height:16px;border-radius:4px;}
.run {background:#10B981;} .boot {background:#F59E0B;animation:blink 0.8s infinite;}
@keyframes blink {50% {opacity:.35;}}
.reason {margin:4px 0;}
.mono {font-family: ui-monospace, Consolas, monospace; font-size:.95rem; line-height:1.7}
.badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600;color:white}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>☁️ AI-Driven Cloud Resource Allocation & Predictive Autoscaling</h1>
<p>Forecasts CPU, memory and traffic with Machine Learning, scales Kubernetes pods <b>before</b> the spike,
learns the cost-vs-SLA trade-off with Reinforcement Learning, detects anomalies and explains every decision.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    source = st.radio("Workload data", ["Built-in scenario", "Upload CSV"], horizontal=True)
    uploaded = None
    if source == "Built-in scenario":
        scenario_label = st.selectbox("Scenario", list(SCENARIOS.keys()))
        seed = st.number_input("Random seed", 0, 9999, 42)
    else:
        uploaded = st.file_uploader("CSV: column `cpu` (vCPU) required; `requests`, `memory` optional. "
                                    "One row = 5 min, at least 8 days.", type="csv")
    st.subheader("☸️ Cluster")
    pod_cpu = st.select_slider("Pod size: vCPU", [1, 2, 4, 8], value=4)
    pod_mem = st.select_slider("Pod size: memory (GiB)", [2, 4, 8, 16], value=8)
    price_hr = st.number_input("Price per pod-hour ($)", 0.02, 5.0, 0.20, 0.01)
    boot_min = st.select_slider("Scale-up delay (node + pod start, min)", [5, 10, 15, 20], value=10)
    st.subheader("🎯 SLA requirements")
    max_rt = st.slider("Maximum response time (ms)", 200, 1000, 500, 50)
    avail_target = st.slider("Target availability (%)", 95.0, 99.9, 99.0, 0.1)
    max_util = st.slider("Maximum resource utilization (%)", 60, 95, 80, 5) / 100
    st.subheader("🤖 AI")
    sla_weight = st.slider("SLA penalty weight", 2, 40, 10,
                           help="Cost of one dropped vCPU of work, as a multiple of its price. "
                                "Higher = the RL agent keeps more headroom.")
    rl_episodes = st.slider("RL training episodes", 10, 80, 40, 10)
    run_clicked = st.button("🚀 Train AI & run simulation", type="primary", width="stretch")


@st.cache_data(show_spinner=False)
def load_scenario(scenario, seed):
    return generate_workload(scenario=scenario, seed=seed)


@st.cache_resource(show_spinner=False)
def result_store():
    """Trained results shared by all sessions, keyed by data + settings."""
    return {}


@st.cache_data(show_spinner=False, max_entries=40)
def cached_what_if(key, pct, pc):
    res = what_if(st.session_state.R, pct, pc)
    return {k: (v[0], v[1]) for k, v in res.items()}


# Build the workload
if source == "Built-in scenario":
    wl = load_scenario(SCENARIOS[scenario_label], int(seed))
    data_name = scenario_label
    data_key = f"{SCENARIOS[scenario_label]}-{seed}"
else:
    if uploaded is None:
        st.info("⬅️ Upload a CSV, or switch back to a built-in scenario. A sample file can be downloaded "
                "from the **Architecture** tab.")
        st.stop()
    raw = pd.read_csv(uploaded)
    if "cpu" not in raw.columns or len(raw) < 8 * STEPS_PER_DAY:
        st.error(f"The CSV needs a numeric `cpu` column and at least {8 * STEPS_PER_DAY} rows (8 days of 5-min data).")
        st.stop()
    cpu_u = raw["cpu"].astype(float).clip(lower=0.1).to_numpy()
    wl = pd.DataFrame({
        "time": pd.date_range("2026-01-05", periods=len(raw), freq="5min"),
        "cpu": cpu_u,
        "requests": raw["requests"].astype(float).to_numpy() if "requests" in raw else cpu_u * REQ_PER_VCPU,
        "memory": raw["memory"].astype(float).to_numpy() if "memory" in raw else 10 + 0.45 * cpu_u,
        "injected": raw["injected"].fillna("").astype(str).to_numpy() if "injected" in raw else "",
    })
    data_name = f"Uploaded: {uploaded.name}"
    data_key = f"upload-{uploaded.name}-{len(raw)}-{float(cpu_u.sum()):.3f}"

cfg = dict(pod_cpu=pod_cpu, pod_mem=pod_mem, price_hr=float(price_hr), boot_steps=boot_min // STEP_MIN,
           max_rt=max_rt, avail_target=float(avail_target), max_util=float(max_util),
           sla_weight=float(sla_weight), rl_episodes=int(rl_episodes))
key = f"{data_key}|{sorted(cfg.items())}"
if "key" not in st.session_state or run_clicked:
    st.session_state.key, st.session_state.wl, st.session_state.cfg = key, wl, cfg
    st.session_state.data_name = data_name
if st.session_state.key != key:
    st.warning("Settings changed. Press **🚀 Train AI & run simulation** in the sidebar to apply them.")

store = result_store()
if st.session_state.key not in store:
    bar = st.progress(0.0, text="Starting ...")
    store[st.session_state.key] = run_experiment(
        st.session_state.wl, progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg), **st.session_state.cfg)
    bar.empty()
    while len(store) > 6:
        store.pop(next(iter(store)))
R = store[st.session_state.key]
st.session_state.R = R

C = R["cfg"]
wl = R["wl"]
runs, S = R["runs"], R["summary"]
split, n = R["split"], R["n"]
times = wl["time"]
tt = times.iloc[split:n].reset_index(drop=True)          # test-period timestamps
days_lbl = list(dict.fromkeys(tt.dt.strftime("%a %d %b")))
H = runs[HYBRID]


def tlabel(t):
    return times.iloc[int(t)].strftime("%a %d %b, %H:%M")


def kpi(col, value, label, color):
    col.markdown(f'<div class="kpi"><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


tabs = st.tabs(["🏠 Overview", "🔮 Prediction", "📈 Autoscaling", "🧠 Explainable AI", "⚠️ Anomalies",
                "💰 Optimization & Cost", "🧪 What-If", "🏗️ Architecture"])

# ===========================================================================
# 1. OVERVIEW
# ===========================================================================
with tabs[0]:
    tot = S["Cost + penalty ($)"]
    st.caption(f"Data: **{st.session_state.data_name}** · AI trained on past data; every number is measured on "
               f"the final **{R['days']:.0f} unseen days** · pod = {C['pod_cpu']} vCPU / {C['pod_mem']} GiB "
               f"@ ${C['price_hr']:.2f}/h · SLA: ≤ {C['max_rt']} ms, ≥ {C['avail_target']}% availability, "
               f"≤ {100 * C['max_util']:.0f}% utilization")
    c = st.columns(4)
    kpi(c[0], f"{100 * (1 - tot[HYBRID] / tot[STATIC]):.0f}%", "lower total cost vs static over-provisioning", "#10B981")
    kpi(c[1], f"{100 * (1 - tot[HYBRID] / tot[REACTIVE]):.0f}%", "lower total cost vs reactive Kubernetes HPA", "#3B82F6")
    b_r, b_h = S.loc[REACTIVE, "SLA breaches (%)"], S.loc[HYBRID, "SLA breaches (%)"]
    kpi(c[2], f"{100 * (1 - b_h / b_r):.0f}%" if b_r > 0 else "-", "fewer SLA breaches than reactive HPA", "#8B5CF6")
    kpi(c[3], f"${(S.loc[STATIC, 'Monthly cost ($)'] - S.loc[HYBRID, 'Monthly cost ($)']) * 12:,.0f}",
        "projected yearly saving vs static", "#F59E0B")

    st.subheader("Cluster status: latest interval (AI Forecast+RL)")
    last = H.iloc[-1]
    rt = float(last["response_ms"])
    c = st.columns(6)
    c[0].metric("Request rate", f"{R['req'][n - 1]:,.0f} req/s")
    c[1].metric("CPU utilization", f"{100 * last['cpu_util']:.0f}%")
    c[2].metric("Memory utilization", f"{100 * min(last['mem_util'], 1):.0f}%")
    c[3].metric("Pods", f"{int(last['pods_running'])}", f"+{int(last['pods_starting'])} starting"
                if last["pods_starting"] else None)
    c[4].metric("Response time", f"{rt:,.0f} ms")
    c[5].metric("SLA status", "✅ Healthy" if not last["sla_breach"] else "❌ Breach")

    last_day = slice(len(H) - STEPS_PER_DAY, len(H))
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Requests / s", "CPU vs capacity (vCPU)", "Memory vs capacity (GiB)"))
    x = tt.iloc[last_day]
    fig.add_scatter(x=x, y=R["req"][split:n][last_day], line=dict(color="#6366F1"), row=1, col=1, name="Requests")
    fig.add_scatter(x=x, y=H["cpu"].iloc[last_day], line=dict(color=INK), row=1, col=2, name="CPU used")
    fig.add_scatter(x=x, y=H["capacity"].iloc[last_day], line=dict(color="#10B981", shape="hv"), row=1, col=2, name="CPU capacity")
    fig.add_scatter(x=x, y=H["memory"].iloc[last_day], line=dict(color="#EC4899"), row=1, col=3, name="Memory used")
    fig.add_scatter(x=x, y=H["mem_capacity"].iloc[last_day], line=dict(color="#10B981", shape="hv", dash="dot"),
                    row=1, col=3, name="Memory capacity")
    fig.update_layout(height=280, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("SLA compliance by autoscaler (test period)")
    rows = []
    for p in ORDER:
        rows.append({"Autoscaler": SHORT[p],
                     "Availability": f"{S.loc[p, 'Availability (%)']:.2f}% "
                                     f"{'✅' if S.loc[p, 'Availability (%)'] >= C['avail_target'] else '❌'}",
                     "p95 response time": f"{S.loc[p, 'p95 response (ms)']:,.0f} ms "
                                          f"{'✅' if S.loc[p, 'p95 response (ms)'] <= C['max_rt'] else '❌'}",
                     "Avg CPU utilization": f"{S.loc[p, 'Avg CPU util (%)']:.0f}%",
                     "SLA breaches (5-min intervals)": f"{S.loc[p, 'SLA breaches (%)']:.1f}%",
                     "Monthly cost": f"${S.loc[p, 'Monthly cost ($)']:,.0f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Autoscaler"), width="stretch")

# ===========================================================================
# 2. PREDICTION
# ===========================================================================
with tabs[1]:
    st.markdown(f"""
**Model:** Gradient-Boosted Decision Trees (scikit-learn `HistGradientBoostingRegressor`), one per metric.
It predicts the **peak value over the next {R['horizon'] * STEP_MIN} minutes** (the time a new pod needs to
start) from past data only: recent values, same time yesterday and last week, rolling averages, trend and
time of day. A **conformal prediction interval** is calibrated on a held-out week so the upper bound covers
the true peak about 95% of the time. The autoscaler provisions for that bound, not the average.
""")
    metric = st.radio("Metric", ["CPU (vCPU)", "Memory (GiB)", "Requests (req/s)"], horizontal=True)
    mkey = {"CPU (vCPU)": "cpu", "Memory (GiB)": "memory", "Requests (req/s)": "requests"}[metric]
    F = R["forecasts"][mkey]
    m = F["metrics"]
    c = st.columns(4)
    c[0].metric("Forecast error (MAPE)", f"{m['MAPE (%)']:.1f}%",
                f"{m['MAPE (%)'] - m['Naive MAPE (%)']:+.1f} pts vs naive", delta_color="inverse",
                help="Naive baseline = 'the next value equals the last one'")
    c[1].metric("Mean absolute error", f"{m['MAE']:,.1f}")
    c[2].metric("Interval coverage", f"{m['Coverage (%)']:.0f}%", help="How often the real peak stayed below the upper bound")
    c[3].metric("Safety margin", f"+{F['offset']:,.1f}", help="Conformal margin learned on the calibration week")
    h = R["horizon"]
    day_pick = st.select_slider("Show", ["Whole test week"] + days_lbl, value="Whole test week", key="pred_zoom")
    idx = np.arange(split, n - h)
    if day_pick != "Whole test week":
        idx = idx[(times.iloc[idx].dt.strftime("%a %d %b") == day_pick).to_numpy()]
    fig = go.Figure()
    fig.add_scatter(x=times.iloc[idx], y=F["upper"][idx], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=times.iloc[idx], y=F["median"][idx], fill="tonexty", fillcolor="rgba(59,130,246,0.18)",
                    line=dict(color="#3B82F6", width=1.6), name="AI forecast (band = 95% upper bound)")
    fig.add_scatter(x=times.iloc[idx], y=F["actual"][idx], line=dict(color=INK, width=1.1), name="Actual")
    fig.update_layout(height=420, title=f"Actual vs predicted {metric} on unseen data", hovermode="x unified", **PLOT)
    st.plotly_chart(fig, width="stretch")
    if st.checkbox("Explain the model: which inputs matter most (permutation importance)"):
        with st.spinner("Computing ..."):
            fi = F["model"].feature_importance(split, n)
        fig = px.bar(fi.tail(10), x="importance", y="feature", orientation="h", color_discrete_sequence=["#3B82F6"])
        fig.update_layout(height=380, xaxis_title="Increase in error when this input is scrambled", yaxis_title="",
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 3. AUTOSCALING
# ===========================================================================
with tabs[2]:
    st.markdown("Compare how each autoscaler matches capacity to demand. **Reactive HPA** waits for high CPU and "
                "is late; the **AI** scales before the spike.")
    pick = st.multiselect("Autoscalers", ORDER, default=[REACTIVE, HYBRID], format_func=lambda p: SHORT[p])
    zoom = st.select_slider("Zoom", ["Whole test week"] + days_lbl, value=days_lbl[min(3, len(days_lbl) - 1)], key="as_zoom")
    mask = np.ones(len(tt), bool) if zoom == "Whole test week" else (tt.dt.strftime("%a %d %b") == zoom).to_numpy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                        subplot_titles=("CPU demand vs provisioned capacity (vCPU)", "Pod replicas"))
    fig.add_scatter(x=tt[mask], y=H["cpu"][mask], name="CPU demand", line=dict(color=INK, width=1.6), row=1, col=1)
    for p in pick:
        d = runs[p]
        fig.add_scatter(x=tt[mask], y=d["capacity"][mask], name=f"{SHORT[p]}", line=dict(color=COLORS[p], width=2, shape="hv"),
                        row=1, col=1, legendgroup=p)
        miss = mask & d["sla_breach"].to_numpy()
        if miss.any():
            fig.add_scatter(x=tt[miss], y=d["cpu"][miss], mode="markers", name=f"{SHORT[p]} SLA breach", legendgroup=p,
                            marker=dict(color=COLORS[p], size=7, symbol="x", line=dict(width=1, color="#EF4444")), row=1, col=1)
        fig.add_scatter(x=tt[mask], y=d["pods"][mask], line=dict(color=COLORS[p], width=2, shape="hv"), showlegend=False,
                        legendgroup=p, row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", **PLOT)
    st.plotly_chart(fig, width="stretch")

    ev = R["decisions"]
    ups, downs = (ev["action"] == "Scale Up").sum(), (ev["action"] == "Scale Down").sum()
    c = st.columns(4)
    c[0].metric("AI scale-up events", int(ups))
    c[1].metric("AI scale-down events", int(downs))
    c[2].metric("Reactive HPA SLA breaches", f"{S.loc[REACTIVE, 'SLA breaches (%)']:.1f}%")
    c[3].metric("AI SLA breaches", f"{S.loc[HYBRID, 'SLA breaches (%)']:.1f}%")

    st.divider()
    st.subheader("🎬 Live replay")
    st.caption("Replays one day, 5 minutes per frame. 🟩 running pod · 🟨 starting pod (billed, not yet serving).")
    a, b, c3 = st.columns([2, 2, 1])
    live_day = a.selectbox("Day", days_lbl, index=min(3, len(days_lbl) - 1))
    other = b.selectbox("Compare AI with", [REACTIVE, STATIC, PREDICT], format_func=lambda p: SHORT[p])
    speed = c3.select_slider("Speed", ["slow", "normal", "fast"], value="normal")
    go_btn = st.button("▶️ Start live replay", type="primary")
    idx_day = np.where((tt.dt.strftime("%a %d %b") == live_day).to_numpy())[0]
    players = [other, HYBRID]
    holders = [col.empty() for col in st.columns(2)]
    chart_ph = st.empty()
    ymax = 1.12 * max(runs[p]["capacity"].iloc[idx_day].max() for p in players + [HYBRID])
    ymax = max(ymax, 1.12 * H["cpu"].iloc[idx_day].max())

    def panel(df, p):
        row = df.iloc[-1]
        run_n, st_n = int(row["pods_running"]), int(row["pods_starting"])
        load = row["cpu"] / max(row["capacity"], 1)
        colr = "#EF4444" if load > 1 else ("#F59E0B" if load > C["max_util"] else "#10B981")
        return f"""<div class="card" style="border-top:5px solid {COLORS[p]}">
          <b style="font-size:1.05rem">{SHORT[p]}</b><span style="float:right;opacity:.7">{run_n} running · {st_n} starting</span>
          <div class="vmgrid" style="margin:10px 0">{'<div class="vm run"></div>' * run_n}{'<div class="vm boot"></div>' * st_n}</div>
          <div style="font-size:.85rem">CPU load {100 * load:.0f}% · response {row['response_ms']:,.0f} ms</div>
          <div style="background:rgba(128,128,128,.2);border-radius:6px;height:10px">
            <div style="width:{min(100, 100 * load):.0f}%;background:{colr};height:10px;border-radius:6px"></div></div>
          <div style="display:flex;justify-content:space-between;margin-top:10px">
            <span>💰 <b>${df['cost'].sum():,.2f}</b></span><span>⚠️ <b>{int(df['sla_breach'].sum())}</b> SLA breaches</span>
            <span>✅ <b>{100 * df['served'].sum() / df['cpu'].sum():.2f}%</b> served</span></div></div>"""

    def draw(k):
        sl = idx_day[: k + 1]
        for hd, p in zip(holders, players):
            hd.markdown(panel(runs[p].iloc[sl], p), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_scatter(x=tt[sl], y=H["cpu"].iloc[sl], name="CPU demand", line=dict(color=INK, width=2))
        for p in players:
            fig.add_scatter(x=tt[sl], y=runs[p]["capacity"].iloc[sl], name=SHORT[p], line=dict(color=COLORS[p], width=2.5, shape="hv"))
        fig.update_layout(height=300, title=f"{tt[sl[-1]]:%a %d %b  %H:%M}", yaxis_title="vCPU",
                          xaxis_range=[tt[idx_day[0]], tt[idx_day[-1]]], yaxis_range=[0, ymax], **PLOT)
        chart_ph.plotly_chart(fig, width="stretch", key=f"live_{k}_{time.time()}")

    if go_btn:
        stride, delay = {"slow": (1, 0.10), "normal": (2, 0.03), "fast": (6, 0.0)}[speed]
        for k in range(0, len(idx_day), stride):
            draw(k)
            time.sleep(delay)
    draw(len(idx_day) - 1)

# ===========================================================================
# 4. EXPLAINABLE AI
# ===========================================================================
with tabs[3]:
    st.markdown("Every scaling decision made by the AI comes with a **human-readable reason**: the forecast, "
                "the SLA risk, the anomaly signal and the buffer chosen by the RL agent.")
    ev = R["decisions"].copy()
    ev["time"] = ev["t"].map(tlabel)
    ev["day"] = ev["t"].map(lambda t: times.iloc[int(t)].strftime("%a %d %b"))
    c = st.columns([1, 1, 2])
    f_action = c[0].selectbox("Action", ["All", "Scale Up", "Scale Down"])
    f_day = c[1].selectbox("Day", ["All"] + days_lbl)
    big_only = c[2].checkbox("Only big changes (±3 pods or more)", value=True)
    view = ev
    if f_action != "All":
        view = view[view["action"] == f_action]
    if f_day != "All":
        view = view[view["day"] == f_day]
    if big_only:
        view = view[(view["to"] - view["from"]).abs() >= 3]
    if view.empty:
        st.info("No decisions match these filters.")
    else:
        opts = view.index.tolist()
        default = int(view.assign(d=(view["to"] - view["from"]).abs())["d"].idxmax())
        sel = st.selectbox("Decision", opts, index=opts.index(default),
                           format_func=lambda i: f"{ev.loc[i, 'time']}  ·  {ev.loc[i, 'action']}: "
                                                 f"{ev.loc[i, 'from']} → {ev.loc[i, 'to']} pods")
        d = ev.loc[sel]
        up = d["action"] == "Scale Up"
        col = "#10B981" if up else "#3B82F6"
        left, right = st.columns([3, 2])
        with left:
            reasons = "".join(f'<div class="reason">• {r}</div>' for r in d["reasons"])
            st.markdown(f"""<div class="card" style="border-left:6px solid {col}">
              <div style="opacity:.7">{d['time']}</div>
              <div style="font-size:1.6rem;font-weight:700;color:{col}">{'⬆️' if up else '⬇️'} {d['action']}: {d['from']} → {d['to']} replicas</div>
              <div style="margin-top:10px;font-weight:600">Reason</div>{reasons}</div>""", unsafe_allow_html=True)
        with right:
            st.markdown(f"""<div class="card mono">
              <b>How the number was computed</b><br>
              predicted peak CPU = {d['pred_cpu']:.1f} vCPU<br>
              RL target utilization = {100 * d['target']:.0f}%<br>
              pods by CPU = ⌈{d['pred_cpu']:.1f} / ({C['pod_cpu']} × {d['target']:.2f})⌉ = <b>{d['by_cpu']}</b><br>
              predicted memory = {d['pred_mem']:.1f} GiB<br>
              pods by memory = ⌈{d['pred_mem']:.1f} / ({C['pod_mem']} × 0.90)⌉ = <b>{d['by_mem']}</b><br>
              recommendation = max(CPU, memory) = <b>{max(d['by_cpu'], d['by_mem'])}</b><br>
              after HPA-style stabilization → <b>{d['to']}</b> pods
            </div>""", unsafe_allow_html=True)
        with st.expander(f"All {len(view)} matching decisions"):
            st.dataframe(view[["time", "action", "from", "to", "pred_cpu", "target"]]
                         .rename(columns={"pred_cpu": "predicted CPU", "target": "RL target util"})
                         .style.format({"predicted CPU": "{:.1f}", "RL target util": "{:.0%}"}),
                         width="stretch", hide_index=True)

    st.divider()
    st.subheader("What the RL agent learned")
    st.markdown("""
| | |
|---|---|
| **Algorithm** | Tabular Q-learning with ε-greedy exploration (ε 1.0 → 0.02) |
| **State** | time of day · demand trend · volatility · was capacity saturated? |
| **Action** | target utilization applied to the ML forecast (how much safety headroom to keep) |
| **Reward** | −(pod cost + SLA penalty for dropped work and slow responses) |
""")
    l, r = st.columns(2)
    hist = R["rl_history"]
    with l:
        fig = go.Figure(go.Scatter(x=hist["episode"], y=-hist["reward"], mode="lines+markers", line=dict(color="#10B981")))
        fig.update_layout(title="Learning curve: penalised cost per episode (lower = smarter)", xaxis_title="Episode",
                          yaxis_title="Cost (pod-steps + penalty)", height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with r:
        tbl = R["rl"].policy_table()
        fig = px.imshow(tbl.values, x=tbl.columns, y=tbl.index, text_auto=".0f", aspect="auto",
                        color_continuous_scale="RdYlGn", zmin=100 * min(R["targets"]), zmax=100 * max(R["targets"]),
                        labels=dict(color="Target %"))
        fig.update_layout(title="Learned policy: target utilization (red = more headroom)", xaxis_title="Hour of day",
                          yaxis_title="Volatility", height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 5. ANOMALIES
# ===========================================================================
with tabs[4]:
    A = R["anomalies"].copy()
    st.markdown("""**Model:** Robust Covariance anomaly detector (Minimum Covariance Determinant, scikit-learn
`EllipticEnvelope`). It learns the normal relationship between **sudden traffic change**, **traffic vs last week**,
**memory not explained by traffic** and **CPU per request**, then flags points that are statistically far from
normal (Mahalanobis distance). Consecutive points are grouped into events with a type and severity.""")
    c = st.columns(4)
    c[0].metric("Anomaly events", len(A))
    for i, sev in enumerate(["High", "Medium", "Low"]):
        c[i + 1].metric(f"{sev} severity", int((A["severity"] == sev).sum()) if len(A) else 0)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Requests / s", "CPU (vCPU)", "Memory (GiB)"))
    for i, (y, colr) in enumerate([(R["req"], "#6366F1"), (R["cpu"], INK), (R["mem"], "#EC4899")]):
        fig.add_scatter(x=tt, y=y[split:n], line=dict(color=colr, width=1), row=i + 1, col=1, showlegend=False)
    for _, e in A.iterrows():
        for row in (1, 2, 3):
            fig.add_vrect(x0=times.iloc[e["start_t"]] - pd.Timedelta(minutes=5), x1=times.iloc[e["end_t"]] + pd.Timedelta(minutes=5),
                          fillcolor=SEV_COLORS[e["severity"]], opacity=0.25, line_width=0, row=row, col=1)
    fig.update_layout(height=560, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch")
    if len(A):
        sig = {"Traffic spike": "traffic", "Traffic drop": "traffic", "Memory spike": "memory", "Abnormal CPU": "CPU per request"}
        table = pd.DataFrame({
            "Timestamp": A["start_t"].map(tlabel),
            "Type": A["type"],
            "Severity": A["severity"],
            "Duration (min)": A["duration_min"],
            "Anomaly score": A["score"].round(1),
            "Why": [f"{sig.get(tp, tp)} signal {max(abs(a), abs(b), abs(c_)):.0f}σ from normal"
                    for tp, a, b, c_ in zip(A["type"], A["jump_z"].abs().combine(A["traffic_z"].abs(), max),
                                            A["memory_z"], A["cpu_eff_z"])],
        })
        st.dataframe(table.style.map(lambda v: f"color:{SEV_COLORS.get(v, 'inherit')};font-weight:600",
                                     subset=["Severity"]), width="stretch", hide_index=True)
    ae = R["anomaly_eval"]
    if len(ae):
        st.subheader("Validation against known (injected) anomalies")
        st.caption("The built-in scenarios contain hidden anomalies the model never saw labelled: a runaway process "
                   "(CPU up, traffic flat), a viral traffic spike and a memory leak. Other events found are genuine "
                   "unusual bursts in the data.")
        ae = ae.copy()
        ae["When"] = ae["start_t"].map(tlabel)
        st.dataframe(ae[["When", "Injected anomaly", "Detected", "Detected as", "Delay (min)"]], width="stretch", hide_index=True)

# ===========================================================================
# 6. OPTIMIZATION & COST
# ===========================================================================
with tabs[5]:
    st.markdown("**Current vs recommended resources.** *Current* = a typical setup with a reactive Kubernetes HPA. "
                "*Recommended* = what the AI engine allocates.")
    cur, rec = runs[REACTIVE], H
    mem_per_pod = (rec["memory"] / rec["pods_running"]).quantile(0.99)
    mem_req = math.ceil(mem_per_pod * 1.25 * 2) / 2
    cpu_per_pod = (rec["cpu"] / rec["pods_running"]).quantile(0.99)
    comp = pd.DataFrame({
        "Current (Reactive HPA)": [f"{cur['pods'].mean():.1f}", f"{int(cur['pods'].max())}",
                                   f"{C['pod_cpu']} vCPU", f"{C['pod_mem']} GiB",
                                   f"{S.loc[REACTIVE, 'Avg CPU util (%)']:.0f}%", f"{S.loc[REACTIVE, 'Avg memory util (%)']:.0f}%",
                                   f"{S.loc[REACTIVE, 'SLA breaches (%)']:.1f}%", f"${S.loc[REACTIVE, 'Monthly cost ($)']:,.0f}"],
        "AI Recommended": [f"{rec['pods'].mean():.1f}", f"{int(rec['pods'].max())}",
                           f"{C['pod_cpu']} vCPU (p99 use {cpu_per_pod:.1f})", f"{mem_req:g} GiB (p99 use {mem_per_pod:.1f})",
                           f"{S.loc[HYBRID, 'Avg CPU util (%)']:.0f}%", f"{S.loc[HYBRID, 'Avg memory util (%)']:.0f}%",
                           f"{S.loc[HYBRID, 'SLA breaches (%)']:.1f}%", f"${S.loc[HYBRID, 'Monthly cost ($)']:,.0f}"],
    }, index=["Average replicas", "Peak replicas", "CPU request per pod", "Memory request per pod",
              "Avg CPU utilization", "Avg memory utilization", "SLA breaches", "Monthly cost"])
    l, r = st.columns([3, 2])
    with l:
        st.dataframe(comp, width="stretch")
        if mem_req < C["pod_mem"]:
            st.success(f"💡 **Right-sizing tip (VPA):** pods never use more than ~{mem_per_pod:.1f} GiB. Lowering the memory "
                       f"request from {C['pod_mem']} GiB to {mem_req:g} GiB frees "
                       f"{(C['pod_mem'] - mem_req) * rec['pods'].mean():,.0f} GiB of reserved memory across the cluster.")
    with r:
        c = st.columns(2)
        c[0].metric("Current monthly cost", f"${S.loc[REACTIVE, 'Monthly cost ($)']:,.0f}")
        c[1].metric("Optimized monthly cost", f"${S.loc[HYBRID, 'Monthly cost ($)']:,.0f}",
                    f"-${S.loc[REACTIVE, 'Monthly cost ($)'] - S.loc[HYBRID, 'Monthly cost ($)']:,.0f}", delta_color="inverse")
        c = st.columns(2)
        c[0].metric("vs static over-provisioning", f"${S.loc[STATIC, 'Monthly cost ($)']:,.0f}")
        c[1].metric("Yearly saving vs static", f"${12 * (S.loc[STATIC, 'Monthly cost ($)'] - S.loc[HYBRID, 'Monthly cost ($)']):,.0f}")

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_bar(x=[SHORT[p] for p in ORDER], y=[S.loc[p, "Monthly cost ($)"] for p in ORDER], name="Pod cost",
                    marker_color=[COLORS[p] for p in ORDER])
        fig.add_bar(x=[SHORT[p] for p in ORDER], y=[S.loc[p, "SLA penalty ($)"] * 30 / R["days"] for p in ORDER],
                    name="SLA penalty", marker_color="#EF4444")
        fig.update_layout(barmode="stack", title="Monthly cost = pods + SLA penalty (lower is better)", yaxis_title="$",
                          height=380, **PLOT)
        st.plotly_chart(fig, width="stretch")
    with r:
        fig = go.Figure()
        fig.add_bar(x=[SHORT[p] for p in ORDER], y=[S.loc[p, "Avg CPU util (%)"] for p in ORDER],
                    marker_color=[COLORS[p] for p in ORDER],
                    text=[f"{S.loc[p, 'Avg CPU util (%)']:.0f}%" for p in ORDER], textposition="outside")
        fig.update_layout(title="Average CPU utilization (higher = less waste)", yaxis_range=[0, 100], height=380,
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with st.expander("Full comparison table"):
        full = S.copy()
        full.index = [SHORT[i] for i in full.index]
        st.dataframe(full.style.format("{:,.2f}"), width="stretch")

# ===========================================================================
# 7. WHAT-IF
# ===========================================================================
with tabs[6]:
    st.markdown("Ask *“what happens if…”* **without touching the live cluster**. The AI re-plans the test week "
                "under the new conditions and compares that with keeping today's plan.")
    c = st.columns([3, 2])
    pct = c[0].slider("Traffic change (%)", -50, 200, 30, 10)
    pc = c[1].select_slider("Pod size for the scenario (vCPU)", [1, 2, 4, 8], value=C["pod_cpu"])
    with st.spinner("Simulating ..."):
        W = cached_what_if(st.session_state.key, pct, pc)
    (dfn, sn), (dfk, sk) = W["AI plan for new traffic"], W["Keep current plan"]
    base = S.loc[HYBRID]
    dcost = sn["Monthly cost ($)"] - base["Monthly cost ($)"]
    peak_t = int(np.argmax(H["cpu"].to_numpy()))
    l, r = st.columns([1, 1])
    with l:
        st.markdown(f"""<div class="card mono">
        <b>Scenario: traffic {pct:+d}%{'' if pc == C['pod_cpu'] else f', {pc}-vCPU pods'}</b><br><br>
        Current replicas (avg / peak):&nbsp;&nbsp; {base['Avg pods']:.0f} / {int(base['Peak pods'])}<br>
        Predicted replicas (avg / peak): <b>{sn['Avg pods']:.0f} / {int(sn['Peak pods'])}</b><br>
        At the busiest moment:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {int(H['pods'].iloc[peak_t])} → <b>{int(dfn['pods'].iloc[peak_t])}</b> pods<br>
        Expected CPU utilization:&nbsp;&nbsp;&nbsp;&nbsp; <b>{sn['Avg CPU util (%)']:.0f}%</b><br>
        Expected p95 response time:&nbsp; <b>{sn['p95 response (ms)']:,.0f} ms</b> {'✅' if sn['p95 response (ms)'] <= C['max_rt'] else '❌'}<br>
        Expected availability:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{sn['Availability (%)']:.2f}%</b> {'✅' if sn['Availability (%)'] >= C['avail_target'] else '❌'}<br>
        Estimated cost impact:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{'+' if dcost >= 0 else '-'}${abs(dcost):,.0f}/month</b>
        </div>""", unsafe_allow_html=True)
    with r:
        st.markdown(f"""<div class="card" style="border-left:6px solid #EF4444">
        <b>If you DON'T re-scale</b> (keep today's pod plan under the new traffic)<br><br>
        SLA breaches: <b>{sk['SLA breaches (%)']:.1f}%</b> of the time (AI plan: {sn['SLA breaches (%)']:.1f}%)<br>
        p95 response time: <b>{sk['p95 response (ms)']:,.0f} ms</b><br>
        Availability: <b>{sk['Availability (%)']:.2f}%</b><br>
        Average CPU utilization: <b>{sk['Avg CPU util (%)']:.0f}%</b>
        </div>""", unsafe_allow_html=True)
    day_w = st.selectbox("Day to visualise", days_lbl, index=min(3, len(days_lbl) - 1), key="wi_day")
    mk = (tt.dt.strftime("%a %d %b") == day_w).to_numpy()
    fig = go.Figure()
    fig.add_scatter(x=tt[mk], y=H["cpu"][mk], name="Today's demand", line=dict(color="#9CA3AF", dash="dot"))
    fig.add_scatter(x=tt[mk], y=dfn["cpu"][mk], name=f"Demand {pct:+d}%", line=dict(color=INK, width=1.8))
    fig.add_scatter(x=tt[mk], y=dfk["capacity"][mk], name="Capacity: current plan", line=dict(color="#EF4444", shape="hv"))
    fig.add_scatter(x=tt[mk], y=dfn["capacity"][mk], name="Capacity: AI plan", line=dict(color="#10B981", width=2.5, shape="hv"))
    fig.update_layout(height=400, yaxis_title="vCPU", hovermode="x unified", **PLOT)
    st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 8. ARCHITECTURE
# ===========================================================================
with tabs[7]:
    cal = R["cal_start"]
    fig = go.Figure()
    fig.add_scatter(x=times, y=wl["cpu"], line=dict(color=INK, width=0.8))
    for x0, x1, colr, lab in [(0, cal, "rgba(99,102,241,.10)", "Train forecasters"),
                              (cal, split, "rgba(245,158,11,.14)", "Calibrate + train RL"),
                              (split, n - 1, "rgba(16,185,129,.14)", "TEST (unseen)")]:
        fig.add_vrect(x0=times.iloc[x0], x1=times.iloc[x1], fillcolor=colr, line_width=0,
                      annotation_text=lab, annotation_position="top left")
    fig.update_layout(height=300, title="Data timeline: no test data is ever used for training", yaxis_title="CPU (vCPU)",
                      margin=dict(t=50, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.markdown("""
### How the prototype maps to the production architecture
| Architecture layer | In this prototype | In production |
|---|---|---|
| Users, cloud app, load balancer | Synthetic workload generator (daily/weekly seasonality, bursts, anomalies) or your CSV | Real traffic through an Ingress / ALB |
| Kubernetes cluster & pods | Discrete-event cluster simulator: pod start-up delay, billing, CPU/memory capacity | EKS / Kubernetes |
| Monitoring layer | CPU, memory, request rate, response time, pod count every 5 min | Prometheus + CloudWatch |
| Historical data storage | In-memory time series (pandas) | PostgreSQL / Prometheus TSDB |
| **AI/ML engine: workload & demand forecasting** | Gradient-boosted trees + conformal interval, per metric | same model, served by FastAPI |
| **AI/ML engine: anomaly detection** | Robust Covariance (MCD) on traffic/memory/CPU-efficiency signals | same |
| **Resource optimization engine** | Replicas = max(CPU need, memory need) at target utilization; SLA latency model; cost model | same |
| **Intelligent decision engine** | Q-learning agent picks the safety buffer; HPA-style stabilization | writes to HPA/KEDA or the EKS API |
| Kubernetes / AWS actuators | Simulator applies scale up / down | `kubectl scale`, HPA, EC2 Auto Scaling |
| Performance check & feedback loop | Cost, SLA breaches, response time, utilization; RL reward | Same metrics, model retraining on new data |
| What-if · Explainable AI · Cost estimation | Dedicated dashboard tabs | same |

### Autoscalers compared
* **Static:** enough pods for the historical peak + 10%, always on.
* **Reactive HPA:** +2 pods when CPU > max utilization, −1 pod when CPU < 40%, 15-min cooldown (default-style Kubernetes HPA / AWS step scaling).
* **AI Forecast:** provisions for the forecast upper bound at the maximum-utilization target.
* **AI Forecast+RL:** same forecast; the RL agent decides how much headroom to keep in each situation.

**Why the AI wins:** a new pod needs minutes to become ready. A reactive scaler only notices a spike after it starts.
The forecaster sees it coming from daily and weekly patterns and trend, and the RL agent learns when it is safe to
run hot and when to keep extra headroom.
""")
    st.download_button("⬇️ Download this workload as CSV (also the upload format)",
                       wl.to_csv(index=False).encode(), "workload.csv", "text/csv")
