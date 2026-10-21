"""
CloudMind (simple version) - AI Cloud Resource Allocation
Run:  streamlit run app_v2.py
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.simulator import generate_workload, SCENARIOS, STEPS_PER_DAY
from core.pipeline import run_experiment, HYBRID

st.set_page_config(page_title="CloudMind - AI Cloud Resource Allocation", page_icon="☁️", layout="wide")

STATIC, NORMAL, AI = "Static (peak provisioning)", "Reactive (HPA threshold)", HYBRID
NAME = {STATIC: "Fixed servers", NORMAL: "Normal autoscaling", AI: "AI autoscaling"}
COLOR = {STATIC: "#9CA3AF", NORMAL: "#F59E0B", AI: "#10B981"}
ORDER = [STATIC, NORMAL, AI]
INK = "#111827"
WEEK_MIN = 7 * 24 * 60

st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
.hero {background: linear-gradient(120deg,#4F46E5 0%,#0EA5E9 60%,#10B981 100%);
       color:white;padding:20px 28px;border-radius:16px;margin-bottom:14px;}
.hero h1 {color:white;margin:0;font-size:1.9rem;}
.hero p {margin:6px 0 0 0;opacity:.93;font-size:1.05rem;}
.kpi {border-radius:14px;padding:16px 18px;border:1px solid rgba(128,128,128,.25);height:100%;}
.kpi .v {font-size:2rem;font-weight:700;line-height:1.2}
.kpi .l {font-size:.9rem;opacity:.75;}
.step {border-radius:14px;padding:16px 18px;border:1px solid rgba(128,128,128,.25);height:100%;}
.step h4 {margin:0 0 6px 0;}
.card {border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:14px 16px;}
.grid {display:flex;flex-wrap:wrap;gap:4px;min-height:44px;margin:10px 0;}
.pod {width:16px;height:16px;border-radius:4px;}
.on {background:#10B981;} .start {background:#F59E0B;animation:blink .8s infinite;}
@keyframes blink {50% {opacity:.35;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>☁️ CloudMind: AI Cloud Resource Allocation</h1>
<p>AI predicts website traffic and adds or removes cloud servers <b>before</b> the traffic arrives:
lower cost, no slow website.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    scenario_label = st.selectbox("Traffic type", list(SCENARIOS.keys()))
    st.caption("The app creates 4 weeks of realistic website traffic. The AI learns from the first 3 weeks "
               "and is tested on the last week, which it has never seen.")


@st.cache_resource(show_spinner=False)
def result_store():
    return {}


store = result_store()
key = SCENARIOS[scenario_label]
if key not in store:
    bar = st.progress(0.0, text="Training the AI ...")
    store[key] = run_experiment(generate_workload(scenario=key),
                                progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg))
    bar.empty()
R = store[key]

runs, S = R["runs"], R["summary"]
split, n = R["split"], R["n"]
tt = R["wl"]["time"].iloc[split:n].reset_index(drop=True)
days = list(dict.fromkeys(tt.dt.strftime("%a %d %b")))
busy_day = days[min(3, len(days) - 1)]


def slow_minutes(p):
    return S.loc[p, "SLA breaches (%)"] / 100 * WEEK_MIN


def kpi(col, value, label, color):
    col.markdown(f'<div class="kpi"><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


tabs = st.tabs(["🏠 Home", "🔮 Prediction", "📈 Autoscaling", "💰 Cost Comparison"])

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("The problem")
    st.markdown("Websites run on **rented cloud servers**. Rent too many and you waste money. Rent too few and "
                "the website becomes slow when traffic rises. Normal autoscaling adds servers only **after** they "
                "are overloaded, and a new server needs ~10 minutes to start, so users suffer in the meantime.")
    st.subheader("How this project solves it")
    c = st.columns(3)
    for col, (title, text) in zip(c, [
        ("1️⃣ Monitor", "Collect server usage every 5 minutes: CPU, memory and website requests."),
        ("2️⃣ Predict (AI)", "A machine-learning model predicts traffic for the next 15 minutes from past patterns."),
        ("3️⃣ Scale early", "Add servers before the rush, remove them when traffic will drop. An RL agent "
                          "learns how much spare capacity is safe."),
    ]):
        col.markdown(f'<div class="step"><h4>{title}</h4>{text}</div>', unsafe_allow_html=True)

    st.subheader("Result on the test week")
    saving = S.loc[NORMAL, "Monthly cost ($)"] - S.loc[AI, "Monthly cost ($)"]
    c = st.columns(4)
    kpi(c[0], f"${S.loc[AI, 'Monthly cost ($)']:,.0f}", "AI monthly cloud bill", "#10B981")
    kpi(c[1], f"${saving:,.0f}", f"saved per month vs normal autoscaling "
                                 f"({100 * saving / S.loc[NORMAL, 'Monthly cost ($)']:.0f}%)", "#3B82F6")
    kpi(c[2], f"{slow_minutes(AI):,.0f} min", f"slow website per week with AI "
                                              f"(normal: {slow_minutes(NORMAL):,.0f} min)", "#8B5CF6")
    kpi(c[3], f"{100 * (1 - S.loc[AI, 'Monthly cost ($)'] / S.loc[STATIC, 'Monthly cost ($)']):.0f}%",
        "cheaper than keeping fixed servers", "#F59E0B")
    st.info("👉 Open the **📈 Autoscaling** tab to see the AI add servers before the traffic arrives.")

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("The AI predicts how much traffic is coming **in the next 15 minutes**. "
                "Black = what really happened, blue = what the AI predicted beforehand.")
    F = R["forecasts"]["requests"]
    acc = 100 - F["metrics"]["MAPE (%)"]
    c = st.columns(3)
    c[0].metric("Prediction accuracy", f"{acc:.0f}%")
    c[1].metric("Model", "Gradient Boosting")
    c[2].metric("Data used to learn", "3 weeks")
    day = st.selectbox("Day", days, index=days.index(busy_day), key="p_day")
    idx = np.arange(split, n - R["horizon"])
    idx = idx[(R["wl"]["time"].iloc[idx].dt.strftime("%a %d %b") == day).to_numpy()]
    x = R["wl"]["time"].iloc[idx]
    fig = go.Figure()
    fig.add_scatter(x=x, y=F["actual"][idx], name="Actual traffic", line=dict(color=INK, width=1.5))
    fig.add_scatter(x=x, y=F["median"][idx], name="AI prediction", line=dict(color="#3B82F6", width=2.5))
    fig.update_layout(height=430, yaxis_title="Website requests per second", hovermode="x unified",
                      margin=dict(t=20, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# AUTOSCALING
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("**Black line** = work the website needs. **Coloured lines** = server capacity rented. "
                "When the black line goes **above** a coloured line, the website is slow (❌). "
                "The **green AI line rises before the rush**; the orange normal line reacts late.")
    day = st.selectbox("Day", days, index=days.index(busy_day), key="a_day")
    mk = (tt.dt.strftime("%a %d %b") == day).to_numpy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.07,
                        subplot_titles=("Work needed vs server capacity", "Number of servers"))
    fig.add_scatter(x=tt[mk], y=runs[AI]["cpu"][mk], name="Work needed", line=dict(color=INK, width=1.8), row=1, col=1)
    for p in [NORMAL, AI]:
        d = runs[p]
        fig.add_scatter(x=tt[mk], y=d["capacity"][mk], name=NAME[p], legendgroup=p,
                        line=dict(color=COLOR[p], width=2.5, shape="hv"), row=1, col=1)
        bad = mk & d["sla_breach"].to_numpy()
        if bad.any():
            fig.add_scatter(x=tt[bad], y=d["cpu"][bad], mode="markers", name=f"{NAME[p]}: slow", legendgroup=p,
                            marker=dict(color="#EF4444", size=8, symbol="x"), row=1, col=1)
        fig.add_scatter(x=tt[mk], y=d["pods"][mk], showlegend=False, legendgroup=p,
                        line=dict(color=COLOR[p], width=2, shape="hv"), row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", margin=dict(t=40, b=10),
                      legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, width="stretch")

    st.subheader("🎬 Live replay")
    st.caption("Plays the selected day like a video. 🟩 server running · 🟨 server starting (paid, not ready yet).")
    go_btn = st.button("▶️ Play", type="primary")
    idx_day = np.where(mk)[0]
    holders = [col.empty() for col in st.columns(2)]

    def panel(p, k):
        d = runs[p].iloc[idx_day[: k + 1]]
        row = d.iloc[-1]
        on, start = int(row["pods_running"]), int(row["pods_starting"])
        load = row["cpu"] / max(row["capacity"], 1)
        colr = "#EF4444" if load > 0.88 else "#10B981"
        status = "❌ Website slow" if row["sla_breach"] else "✅ Website fast"
        return f"""<div class="card" style="border-top:5px solid {COLOR[p]}">
          <b style="font-size:1.1rem">{NAME[p]}</b>
          <span style="float:right">{tt[idx_day[k]]:%H:%M} · {status}</span>
          <div class="grid">{'<div class="pod on"></div>' * on}{'<div class="pod start"></div>' * start}</div>
          <div style="font-size:.85rem">Servers busy: {100 * min(load, 1):.0f}%</div>
          <div style="background:rgba(128,128,128,.2);border-radius:6px;height:10px">
          <div style="width:{min(100, 100 * load):.0f}%;background:{colr};height:10px;border-radius:6px"></div></div>
          <div style="display:flex;justify-content:space-between;margin-top:10px">
          <span>💰 Cost so far: <b>${d['cost'].sum():,.2f}</b></span>
          <span>🐢 Slow: <b>{5 * int(d['sla_breach'].sum())} min</b></span></div></div>"""

    def draw(k):
        for h, p in zip(holders, [NORMAL, AI]):
            h.markdown(panel(p, k), unsafe_allow_html=True)

    if go_btn:
        for k in range(0, len(idx_day), 2):
            draw(k)
            time.sleep(0.04)
    draw(len(idx_day) - 1)

# ---------------------------------------------------------------------------
# COST COMPARISON
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("Three ways to run the same website for the same test week:")
    st.markdown("""
* **Fixed servers:** always rent enough servers for the busiest moment. Safe, but expensive.
* **Normal autoscaling:** today's standard Kubernetes method. Adds servers after they get busy, so it is late.
* **AI autoscaling:** this project. Predicts traffic and scales in advance.
""")
    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(x=[NAME[p] for p in ORDER], y=[S.loc[p, "Monthly cost ($)"] for p in ORDER],
                               marker_color=[COLOR[p] for p in ORDER],
                               text=[f"${S.loc[p, 'Monthly cost ($)']:,.0f}" for p in ORDER], textposition="outside"))
        fig.update_layout(title="Monthly cloud bill (lower is better)", yaxis_title="$", height=380,
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with r:
        fig = go.Figure(go.Bar(x=[NAME[p] for p in ORDER], y=[slow_minutes(p) for p in ORDER],
                               marker_color=[COLOR[p] for p in ORDER],
                               text=[f"{slow_minutes(p):,.0f} min" for p in ORDER], textposition="outside"))
        fig.update_layout(title="Minutes the website was slow in the week (lower is better)", yaxis_title="minutes",
                          height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    table = pd.DataFrame({
        "Monthly cost": [f"${S.loc[p, 'Monthly cost ($)']:,.0f}" for p in ORDER],
        "Average servers": [f"{S.loc[p, 'Avg pods']:.0f}" for p in ORDER],
        "Server usage (avg)": [f"{S.loc[p, 'Avg CPU util (%)']:.0f}%" for p in ORDER],
        "Slow website per week": [f"{slow_minutes(p):,.0f} min" for p in ORDER],
    }, index=[NAME[p] for p in ORDER])
    st.dataframe(table, width="stretch")
    st.success(f"✅ **Conclusion:** AI autoscaling costs **${S.loc[NORMAL, 'Monthly cost ($)'] - S.loc[AI, 'Monthly cost ($)']:,.0f} "
               f"less per month** than normal autoscaling and has **{slow_minutes(NORMAL) - slow_minutes(AI):,.0f} "
               f"fewer slow minutes per week**. Fixed servers are never slow, but cost "
               f"{S.loc[STATIC, 'Monthly cost ($)'] / S.loc[AI, 'Monthly cost ($)']:.1f}× more.")
