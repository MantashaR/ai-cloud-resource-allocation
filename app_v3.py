"""
AI-Driven Intelligent Cloud Resource Allocation & Predictive Autoscaling - easy-to-read dashboard
Run:  streamlit run app_v3.py
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.simulator import generate_workload, SCENARIOS, STEPS_PER_DAY, STEP_MIN
from core.pipeline import run_experiment, what_if, HYBRID

st.set_page_config(page_title="AI Cloud Resource Allocation", page_icon="☁️", layout="wide",
                   initial_sidebar_state="collapsed")

STATIC, REACTIVE, PREDICT = "Static (peak provisioning)", "Reactive (HPA threshold)", "AI Predictive (ML forecast)"
ORDER = [STATIC, REACTIVE, PREDICT, HYBRID]
NAME = {STATIC: "Fixed servers", REACTIVE: "Normal autoscaling", PREDICT: "AI prediction only",
        HYBRID: "Our AI (prediction + learning)"}
ICON = {STATIC: "🧱", REACTIVE: "🐢", PREDICT: "🔮", HYBRID: "🚀"}
COLORS = {STATIC: "#9CA3AF", REACTIVE: "#F59E0B", PREDICT: "#3B82F6", HYBRID: "#10B981"}
GOOD, BAD, INK = "#10B981", "#EF4444", "#6B7280"
PLOT = dict(margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.15),
            hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

SITUATIONS = {
    "🛒 Normal online shop": "E-commerce (daily + weekly pattern)",
    "⚡ Flash sale (sudden rush)": "Flash sale (sudden unseen surge)",
    "📈 App going viral (steady growth)": "Steady growth (startup going viral)",
}
PROBLEM_INFO = {
    "Traffic spike": ("🌊", "Visitors suddenly jumped, like a sale going viral."),
    "Traffic drop": ("📉", "Visitors suddenly dropped, which can mean something upstream is broken."),
    "Memory spike": ("🧠", "Memory kept growing without more visitors. Looks like a memory leak."),
    "Abnormal CPU": ("🔥", "CPU went up but visitors did not. Looks like a stuck or runaway process."),
}

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#4F46E5 0%,#0EA5E9 55%,#10B981 100%); color:white;
       padding:26px 32px; border-radius:20px; margin-bottom:18px; box-shadow:0 10px 30px rgba(79,70,229,.25);}
.hero h1 {color:white; margin:0; font-size:2.1rem; line-height:1.2;}
.hero p {margin:8px 0 0 0; opacity:.95; font-size:1.1rem;}
.stat {border-radius:18px; padding:20px 22px; min-height:150px; color:white; box-shadow:0 6px 18px rgba(0,0,0,.10);
       display:flex; flex-direction:column; justify-content:center;}
.stat .v {font-size:2.5rem; font-weight:800; line-height:1.1;}
.stat .l {font-size:1rem; opacity:.95; margin-top:4px;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:16px; padding:18px 20px; height:100%;
       background:rgba(128,128,128,.04);}
.card h4 {margin:0 0 6px 0;}
.step {text-align:center; border-radius:16px; padding:18px 12px; min-height:215px;
       background:rgba(79,70,229,.07); border:1px solid rgba(79,70,229,.25);}
.step .i {font-size:2.4rem;}
.step .t {font-weight:700; font-size:1.1rem; margin:4px 0;}
.step .d {font-size:.92rem; opacity:.85;}
.arrow {text-align:center; font-size:2rem; padding-top:38px; opacity:.5;}
.note {border-left:5px solid #4F46E5; background:rgba(79,70,229,.07); border-radius:10px;
       padding:12px 16px; margin:6px 0 14px 0; font-size:1.02rem;}
.read {font-size:.9rem; opacity:.8; margin-top:-6px;}
.chip {display:inline-block; padding:2px 12px; border-radius:999px; font-size:.82rem; font-weight:700; color:white;}
.vmgrid {display:flex; flex-wrap:wrap; gap:4px; min-height:44px;}
.vm {width:16px; height:16px; border-radius:4px;}
.run {background:#10B981;} .boot {background:#F59E0B; animation:blink .8s infinite;}
.idle {background:rgba(156,163,175,.55);} .miss {background:transparent; border:2px dashed #EF4444; box-sizing:border-box;}
@keyframes blink {50% {opacity:.35;}}
.big {font-size:1.15rem; line-height:1.7;}
[data-testid="stToolbar"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"] {display:none !important;}
header[data-testid="stHeader"] {background:transparent; height:0;}
.st-key-nav [data-testid="stButtonGroup"] {gap:8px; flex-wrap:wrap;}
.st-key-nav button {font-size:1.02rem !important; font-weight:600; padding:8px 18px !important;
       border-radius:999px !important; min-height:44px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>☁️ AI-Driven Cloud Resource Allocation & Predictive Autoscaling</h1>
<p>An AI that decides <b>how many servers a website needs</b>. It adds servers <b>before</b> the rush,
removes them when it's quiet, and explains every decision.</p>
</div>
""", unsafe_allow_html=True)

top_l, top_r = st.columns([5, 1])
situation = top_l.radio("Pick a situation to test the AI on:", list(SITUATIONS), horizontal=True)
dark = top_r.toggle("🌙 Dark mode", key="dark")
if dark:
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#0F172A !important;}
    .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stSliderTickBar"], .stApp [data-testid="stSliderThumbValue"] {color:#E5E7EB !important;}
    .stApp .card, .stApp .step {background:rgba(255,255,255,.05); border-color:rgba(255,255,255,.15);}
    .stApp .note {background:rgba(129,140,248,.14);}
    .stApp [data-testid="stAlert"] > div {background:rgba(59,130,246,.18) !important;}
    .stApp [data-testid="stExpander"] details {background:rgba(255,255,255,.04); border-color:rgba(255,255,255,.15);}
    .stApp table {background:transparent !important;}
    .stApp th, .stApp td {border-color:rgba(255,255,255,.15) !important;}
    .st-key-nav button, .stApp button[kind="secondary"] {
        background:#1E293B !important; color:#E5E7EB !important; border-color:#334155 !important;}
    .st-key-nav button[kind="segmented_controlActive"] {background:#4F46E5 !important; color:white !important;}
    </style>""", unsafe_allow_html=True)
scenario_label = SITUATIONS[situation]


@st.cache_resource(show_spinner=False)
def result_store():
    """Trained results shared by all sessions, one per scenario."""
    return {}


def train(scenario):
    store = result_store()
    if scenario not in store:
        bar = st.progress(0.0, text="🤖 First run: training the AI ...")
        store[scenario] = run_experiment(
            generate_workload(scenario=scenario, seed=42),
            progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=f"🤖 First run: training the AI. {msg}"))
        bar.empty()
    return store[scenario]


@st.cache_data(show_spinner=False, max_entries=40)
def cached_what_if(scenario, pct):
    res = what_if(result_store()[scenario], pct)
    return {k: (v[0], v[1]) for k, v in res.items()}


R = train(SCENARIOS[scenario_label])

C, runs, S = R["cfg"], R["runs"], R["summary"]
split, n, times = R["split"], R["n"], R["wl"]["time"]
tt = times.iloc[split:n].reset_index(drop=True)
days_lbl = list(dict.fromkeys(tt.dt.strftime("%a %d %b")))
default_day = days_lbl[min(3, len(days_lbl) - 1)]
H, N = runs[HYBRID], runs[REACTIVE]
cost = S["Monthly cost ($)"]
breach = S["SLA breaches (%)"]
ae = R["anomaly_eval"]
found = int((ae["Detected"] == "Yes").sum()) if len(ae) else 0


def chart(fig, where=st, **kw):
    """Plotly chart that follows the dashboard's light/dark switch."""
    if dark:
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    where.plotly_chart(fig, width="stretch", theme=None if dark else "streamlit", **kw)


def stat(col, value, label, color):
    col.markdown(f'<div class="stat" style="background:{color}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def day_mask(day):
    return (tt.dt.strftime("%a %d %b") == day).to_numpy()


def when(t):
    return times.iloc[int(t)].strftime("%a %d %b, %I:%M %p")


PAGES = ["🏠 Start here", "🔮 1. Predict", "⚖️ 2. Scale", "🧠 3. Explain", "⚠️ 4. Problems",
         "💰 5. Money saved", "🧪 Try it yourself", "🏗️ Real-world setup"]
st.session_state.setdefault("nav", PAGES[0])
st.session_state.setdefault("last_page", PAGES[0])


def keep_page():
    # clicking the selected pill would deselect it; keep the last page instead
    if st.session_state.nav is None:
        st.session_state.nav = st.session_state.last_page
    st.session_state.last_page = st.session_state.nav


def go_to(i):
    st.session_state.nav = st.session_state.last_page = PAGES[i]


st.segmented_control("Pages", PAGES, key="nav", on_change=keep_page, label_visibility="collapsed")
page = st.session_state.nav or st.session_state.last_page
st.write("")


def page_buttons(i):
    st.write("")
    st.divider()
    with st.container(key="nav_bottom"):
        b, _, f = st.columns([2, 3, 2])
        if i > 0:
            b.button(f"← {PAGES[i - 1]}", on_click=go_to, args=(i - 1,), width="stretch")
        if i < len(PAGES) - 1:
            f.button(f"Next: {PAGES[i + 1]} →", on_click=go_to, args=(i + 1,), type="primary", width="stretch")

# ===========================================================================
# START HERE
# ===========================================================================
if page == PAGES[0]:
    st.subheader("The problem in one picture 🍽️")
    l, m, r = st.columns(3)
    l.markdown("""<div class="card"><h4>🧱 Fixed servers</h4>
    Like a restaurant that keeps <b>20 waiters every day</b>, even when only 3 tables are full.<br><br>
    ✅ Never slow<br>❌ <b>Wastes a lot of money</b></div>""", unsafe_allow_html=True)
    m.markdown("""<div class="card"><h4>🐢 Normal autoscaling</h4>
    Calls extra waiters <b>only after the queue is already long</b>. They take 10 minutes to arrive.<br><br>
    ✅ Cheaper<br>❌ <b>Customers wait during every rush</b></div>""", unsafe_allow_html=True)
    r.markdown("""<div class="card" style="border:2px solid #10B981"><h4>🚀 Our AI</h4>
    Looks at past patterns, <b>predicts the rush</b> and calls waiters <b>before</b> it starts.<br><br>
    ✅ Cheapest overall<br>✅ <b>Far fewer slow moments</b></div>""", unsafe_allow_html=True)

    st.write("")
    st.subheader(f"Results: {situation}")
    st.caption(f"Measured on {R['days']:.0f} days of data the AI had never seen.")
    c = st.columns(4)
    stat(c[0], f"{100 * (1 - cost[HYBRID] / cost[STATIC]):.0f}%", "cheaper than fixed servers",
         "linear-gradient(135deg,#10B981,#059669)")
    stat(c[1], f"{100 * (1 - cost[HYBRID] / cost[REACTIVE]):.0f}%", "cheaper than normal autoscaling",
         "linear-gradient(135deg,#3B82F6,#2563EB)")
    stat(c[2], f"{100 * (1 - breach[HYBRID] / breach[REACTIVE]):.0f}%" if breach[REACTIVE] > 0 else "-",
         "fewer slow moments than normal autoscaling", "linear-gradient(135deg,#8B5CF6,#6D28D9)")
    stat(c[3], f"{found} / {len(ae)}" if len(ae) else f"{len(R['anomalies'])}",
         "hidden problems caught" if len(ae) else "unusual events spotted",
         "linear-gradient(135deg,#F59E0B,#D97706)")

    st.write("")
    st.subheader("How the AI works, in 4 steps")
    c = st.columns([4, 1, 4, 1, 4, 1, 4])
    for col, (i, t, d) in zip(c[::2], [
        ("👀", "Watch", "Reads CPU, memory and number of visitors every 5 minutes"),
        ("🔮", "Predict", f"Guesses how busy the site will be in the next {R['horizon'] * STEP_MIN} minutes"),
        ("🧠", "Decide", "A learning agent picks how much spare capacity is worth paying for"),
        ("⚙️", "Act & explain", "Adds or removes servers and writes down why"),
    ]):
        col.markdown(f'<div class="step"><div class="i">{i}</div><div class="t">{t}</div>'
                     f'<div class="d">{d}</div></div>', unsafe_allow_html=True)
    for col in c[1::2]:
        col.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)
    st.write("")
    st.info("👉 Click **Next** at the bottom (or the buttons at the top) to go through steps **1 → 5** in order. Each one shows one step with a single chart.")
    page_buttons(0)

# ===========================================================================
# 1. PREDICT
# ===========================================================================
if page == PAGES[1]:
    note(f"🔮 <b>Step 1: predict the future.</b> The AI looks at the recent past, the same time yesterday and the "
         f"same time last week, then guesses how busy the website will be in the <b>next {R['horizon'] * STEP_MIN} "
         f"minutes</b>. That's how long a new server needs to start, so the guess has to come early.")
    what = st.radio("What to predict", ["👥 Visitors", "⚙️ CPU", "🧠 Memory"], horizontal=True)
    mkey = {"👥 Visitors": "requests", "⚙️ CPU": "cpu", "🧠 Memory": "memory"}[what]
    unit = {"requests": "visitors / sec", "cpu": "CPUs needed", "memory": "GB of memory"}[mkey]
    F = R["forecasts"][mkey]
    m = F["metrics"]
    c = st.columns(3)
    stat(c[0], f"{max(0, 100 - m['MAPE (%)']):.0f}%", "accuracy of the AI's guess",
         "linear-gradient(135deg,#3B82F6,#2563EB)")
    stat(c[1], f"{m['Coverage (%)']:.0f} / 100", "rush moments covered by the AI's safe estimate",
         "linear-gradient(135deg,#10B981,#059669)")
    stat(c[2], f"{R['horizon'] * STEP_MIN} min", "advance warning: enough time for a new server to start",
         "linear-gradient(135deg,#8B5CF6,#6D28D9)")
    st.write("")
    day = st.select_slider("Day to show", days_lbl, value=default_day, key=f"pred_day_{scenario_label}")
    h = R["horizon"]
    idx = np.arange(split, n - h)
    idx = idx[(times.iloc[idx].dt.strftime("%a %d %b") == day).to_numpy()]
    fig = go.Figure()
    fig.add_scatter(x=times.iloc[idx], y=F["upper"][idx], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=times.iloc[idx], y=F["median"][idx], fill="tonexty", fillcolor="rgba(59,130,246,0.18)",
                    line=dict(color="#3B82F6", width=2.5), name="🔮 AI's prediction (made in advance)")
    fig.add_scatter(x=times.iloc[idx], y=F["actual"][idx], line=dict(color=INK, width=1.6),
                    name="✔️ What really happened")
    fig.update_layout(height=430, yaxis_title=unit, **PLOT)
    chart(fig)
    st.markdown('<div class="read">📖 <b>How to read:</b> grey line = what really happened · blue line = what the AI '
                'predicted beforehand · light-blue band = the AI\'s safety margin. The closer the lines, '
                'the better.</div>', unsafe_allow_html=True)
    page_buttons(1)

# ===========================================================================
# 2. SCALE
# ===========================================================================
if page == PAGES[2]:
    note("⚖️ <b>Step 2: add or remove servers.</b> The grey area is the work the website needs. Each line is "
         "how much capacity one method provided. If a line drops <b>below</b> the grey area, there weren't "
         "enough servers and the website was <b>slow</b> (marked ✖).")
    def bad(df, dm):
        return int(df["sla_breach"][dm].sum()) + int((df["cpu"][dm] > df["capacity"][dm]).sum())
    gap = {dl: bad(N, day_mask(dl)) - bad(H, day_mask(dl)) for dl in days_lbl}
    best_day = max(gap, key=gap.get)
    day = st.select_slider("Day to show", days_lbl, value=best_day, key=f"scale_day_{scenario_label}")
    st.caption(f"Starts on **{best_day}**, the day with the biggest difference. Slide to see any other day.")
    mk = day_mask(day)
    fig = go.Figure()
    fig.add_scatter(x=tt[mk], y=H["cpu"][mk], fill="tozeroy", fillcolor="rgba(107,114,128,.25)",
                    line=dict(color=INK, width=1), name="Work needed")
    for p in (REACTIVE, HYBRID):
        d = runs[p]
        fig.add_scatter(x=tt[mk], y=d["capacity"][mk], name=f"{ICON[p]} {NAME[p]}",
                        line=dict(color=COLORS[p], width=3, shape="hv"))
        miss = mk & d["sla_breach"].to_numpy()
        if miss.any():
            fig.add_scatter(x=tt[miss], y=d["cpu"][miss], mode="markers", name=f"✖ {NAME[p]}: website slow",
                            marker=dict(color=COLORS[p], size=10, symbol="x", line=dict(width=1.5, color=BAD)))
    fig.update_layout(height=430, yaxis_title="CPUs", **PLOT)
    chart(fig)

    c = st.columns(2)
    for col, p in zip(c, (REACTIVE, HYBRID)):
        d = runs[p][mk]
        slow = int(d["sla_breach"].sum())
        col.markdown(f"""<div class="card" style="border-top:6px solid {COLORS[p]}">
          <h4>{ICON[p]} {NAME[p]}, on {day}</h4>
          <div class="big">🖥️ Average servers running: <b>{d['pods'].mean():.1f}</b><br>
          💰 Cost for the day: <b>${d['cost'].sum():,.2f}</b><br>
          {'🔴' if slow else '🟢'} Slow moments: <b>{slow}</b> × 5 min</div></div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("🎬 Watch it live")
    st.markdown(f"Replays **{day}**, 5 minutes per frame. Each box is one server:  \n"
                "🟩 busy, doing work · ⬜ idle, **wasting money** · 🟥 missing, **website slow** · "
                "🟨 starting up (paid for, but can't help yet)")
    go_btn = st.button("▶️ Play the day", type="primary")
    idx_day = np.where(mk)[0]
    players = [REACTIVE, HYBRID]
    holders = [col.empty() for col in st.columns(2)]
    score_ph = st.empty()
    chart_ph = st.empty()
    ymax = 1.12 * max(max(runs[p]["capacity"].iloc[idx_day].max() for p in players), H["cpu"].iloc[idx_day].max())

    def panel(df, p):
        row = df.iloc[-1]
        run_n, st_n = int(row["pods_running"]), int(row["pods_starting"])
        need = int(np.ceil(row["cpu"] / C["pod_cpu"]))
        busy, idle, missing = min(run_n, need), max(0, run_n - need), max(0, need - run_n)
        load = row["cpu"] / max(row["capacity"], 1)
        colr = BAD if load > 1 else ("#F59E0B" if load > C["max_util"] else GOOD)
        mood = "😫 Overloaded, website slow" if load > 1 else ("😐 Busy" if load > C["max_util"] else "😊 Running smoothly")
        boxes = ('<div class="vm run"></div>' * busy + '<div class="vm idle"></div>' * idle
                 + '<div class="vm miss"></div>' * missing + '<div class="vm boot"></div>' * st_n)
        return f"""<div class="card" style="border-top:6px solid {COLORS[p]}">
          <b style="font-size:1.1rem">{ICON[p]} {NAME[p]}</b>
          <span style="float:right;opacity:.75">{run_n} running · {st_n} starting</span>
          <div class="vmgrid" style="margin:10px 0">{boxes}</div>
          <div style="font-size:.9rem;opacity:.85">🟩 {busy} busy · ⬜ {idle} idle · 🟥 {missing} missing</div>
          <div style="font-size:.95rem">Load {100 * load:.0f}% · {mood}</div>
          <div style="background:rgba(128,128,128,.2);border-radius:6px;height:12px;margin-top:4px">
            <div style="width:{min(100, 100 * load):.0f}%;background:{colr};height:12px;border-radius:6px"></div></div>
          <div style="display:flex;justify-content:space-between;margin-top:10px">
            <span>💰 <b>${df['cost'].sum():,.2f}</b> spent</span>
            <span>🔴 <b>{int(df['sla_breach'].sum())}</b> slow moments</span></div></div>"""

    def draw(k):
        sl = idx_day[: k + 1]
        for hd, p in zip(holders, players):
            hd.markdown(panel(runs[p].iloc[sl], p), unsafe_allow_html=True)
        n_df, a_df = N.iloc[sl], H.iloc[sl]
        saved = n_df["cost"].sum() - a_df["cost"].sum()
        fewer = int(n_df["sla_breach"].sum()) - int(a_df["sla_breach"].sum())
        idle_n = (n_df["pods_running"] - np.ceil(n_df["cpu"] / C["pod_cpu"])).clip(lower=0).sum()
        idle_a = (a_df["pods_running"] - np.ceil(a_df["cpu"] / C["pod_cpu"])).clip(lower=0).sum()
        score_ph.markdown(f"""<div class="card" style="border:2px solid {GOOD};text-align:center;margin:6px 0">
          <span style="font-size:1.15rem">🏆 <b>Our AI vs normal autoscaling, so far today:</b>&nbsp;
          💰 <b>${saved:,.2f}</b> {'cheaper' if saved >= 0 else 'more expensive'} &nbsp;·&nbsp;
          🔴 <b>{abs(fewer)}</b> {'fewer' if fewer >= 0 else 'more'} slow moments &nbsp;·&nbsp;
          ⬜ <b>{(idle_n - idle_a) * STEP_MIN / 60:,.0f}</b> fewer idle server-hours</span></div>""",
                          unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_scatter(x=tt[sl], y=H["cpu"].iloc[sl], name="Work needed", fill="tozeroy",
                        fillcolor="rgba(107,114,128,.25)", line=dict(color=INK, width=1))
        for p in players:
            d = runs[p].iloc[sl]
            fig.add_scatter(x=tt[sl], y=d["capacity"], name=f"{ICON[p]} {NAME[p]}",
                            line=dict(color=COLORS[p], width=3, shape="hv"))
            slow = d["sla_breach"].to_numpy()
            if slow.any():
                fig.add_scatter(x=tt[sl][slow], y=d["cpu"][slow], mode="markers", name=f"✖ {NAME[p]}: slow",
                                marker=dict(color=COLORS[p], size=10, symbol="x", line=dict(width=1.5, color=BAD)))
        fig.update_layout(height=300, title=f"{tt[sl[-1]]:%a %d %b  %I:%M %p}", yaxis_title="CPUs",
                          xaxis_range=[tt[idx_day[0]], tt[idx_day[-1]]], yaxis_range=[0, ymax], **PLOT)
        chart(fig, where=chart_ph, key=f"live_{k}_{time.time()}")

    if go_btn:
        for k in range(0, len(idx_day), 2):
            draw(k)
            time.sleep(0.03)
    draw(len(idx_day) - 1)
    page_buttons(2)

# ===========================================================================
# 3. EXPLAIN
# ===========================================================================
if page == PAGES[3]:
    note("🧠 <b>Step 3: explain every decision.</b> Most AI systems are a black box. This one writes down "
         "<b>why</b> it added or removed servers, so an engineer can check it and trust it.")
    ev = R["decisions"].copy()
    ev["size"] = (ev["to"] - ev["from"]).abs()
    kind = st.radio("Show", ["Biggest decisions", "⬆️ Adding servers", "⬇️ Removing servers"], horizontal=True)
    if kind == "⬆️ Adding servers":
        ev = ev[ev["action"] == "Scale Up"]
    elif kind == "⬇️ Removing servers":
        ev = ev[ev["action"] == "Scale Down"]
    top = ev.sort_values("size", ascending=False).head(6)
    cap_per_pod = C["pod_cpu"]
    cols = st.columns(2)
    for i, (_, d) in enumerate(top.iterrows()):
        up = d["action"] == "Scale Up"
        col = GOOD if up else "#3B82F6"
        diff = abs(int(d["to"]) - int(d["from"]))
        need = d["pred_cpu"]
        have = d["from"] * cap_per_pod
        if up:
            why = [f"🔮 It predicted the website would need <b>{need:.0f} CPUs</b> in the next "
                   f"{R['horizon'] * STEP_MIN} min.",
                   f"🖥️ The {int(d['from'])} running servers only have <b>{have:.0f} CPUs</b>. Not enough.",
                   f"✅ So it added <b>{diff}</b> server{'s' if diff > 1 else ''} early, before visitors felt it."]
        else:
            why = [f"🔮 It predicted the work would drop to <b>{need:.0f} CPUs</b>.",
                   f"🖥️ {int(d['from'])} servers ({have:.0f} CPUs) would mostly sit idle.",
                   f"💰 So it removed <b>{diff}</b> server{'s' if diff > 1 else ''} to save "
                   f"<b>${diff * C['price_hr']:.2f} every hour</b>."]
        why.append(f"🧠 The learning agent chose to keep servers about <b>{100 * d['target']:.0f}% busy</b> "
                   f"in this situation.")
        cols[i % 2].markdown(f"""<div class="card" style="border-left:7px solid {col};margin-bottom:14px">
          <div style="opacity:.7">{when(d['t'])}</div>
          <div style="font-size:1.45rem;font-weight:800;color:{col}">{'⬆️ Added' if up else '⬇️ Removed'} {diff} servers
          <span style="font-size:1rem;opacity:.75">({int(d['from'])} → {int(d['to'])})</span></div>
          <div style="margin-top:8px;line-height:1.75">{'<br>'.join(why)}</div></div>""", unsafe_allow_html=True)
    with st.expander("🔧 Technical details: the AI's raw reasons for these decisions"):
        for _, d in top.iterrows():
            st.markdown(f"**{when(d['t'])}: {d['action']} {d['from']} → {d['to']}**")
            for r in d["reasons"]:
                st.markdown(f"- {r}")
    page_buttons(3)

# ===========================================================================
# 4. PROBLEMS
# ===========================================================================
if page == PAGES[4]:
    note("⚠️ <b>Step 4: spot problems.</b> The AI also learns what <i>normal</i> looks like. When visitors, CPU or "
         "memory behave strangely, it raises an alert. The coloured stripes on the chart are the problems it found.")
    A = R["anomalies"]
    if len(ae):
        c = st.columns(3)
        stat(c[0], f"{found} / {len(ae)}", "problems we hid in the data on purpose, and the AI found",
             "linear-gradient(135deg,#10B981,#059669)")
        stat(c[1], f"{len(A)}", "alerts raised in total", "linear-gradient(135deg,#F59E0B,#D97706)")
        stat(c[2], f"{int((A['severity'] == 'High').sum())}", "high-severity alerts",
             "linear-gradient(135deg,#EF4444,#DC2626)")
        st.write("")
    sev_col = {"High": BAD, "Medium": "#F97316", "Low": "#EAB308"}
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("👥 Visitors per second", "⚙️ CPU used", "🧠 Memory used (GB)"))
    for i, (y, colr) in enumerate([(R["req"], "#6366F1"), (R["cpu"], INK), (R["mem"], "#EC4899")]):
        fig.add_scatter(x=tt, y=y[split:n], line=dict(color=colr, width=1.2), row=i + 1, col=1, showlegend=False)
    for _, e in A.iterrows():
        for row in (1, 2, 3):
            fig.add_vrect(x0=times.iloc[e["start_t"]] - pd.Timedelta(minutes=10),
                          x1=times.iloc[e["end_t"]] + pd.Timedelta(minutes=10),
                          fillcolor=sev_col[e["severity"]], opacity=0.3, line_width=0, row=row, col=1)
    fig.update_layout(height=520, margin=dict(t=40, b=10, l=10, r=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    chart(fig)

    if len(A):
        st.subheader("What it found")
        rank = {"High": 0, "Medium": 1, "Low": 2}
        show = A.assign(r=A["severity"].map(rank)).sort_values(["r", "start_t"]).head(8)
        cols = st.columns(2)
        for i, (_, e) in enumerate(show.iterrows()):
            icon, meaning = PROBLEM_INFO.get(e["type"], ("⚠️", "Unusual behaviour."))
            cols[i % 2].markdown(f"""<div class="card" style="margin-bottom:12px">
              <span style="font-size:1.6rem">{icon}</span> <b style="font-size:1.1rem">{e['type']}</b>
              <span class="chip" style="background:{sev_col[e['severity']]};float:right">{e['severity']}</span>
              <div style="margin-top:6px">{meaning}</div>
              <div style="opacity:.7;font-size:.9rem;margin-top:4px">🕒 {when(e['start_t'])} · lasted {e['duration_min']} min</div>
              </div>""", unsafe_allow_html=True)
    page_buttons(4)

# ===========================================================================
# 5. MONEY SAVED
# ===========================================================================
if page == PAGES[5]:
    yearly = 12 * (cost[REACTIVE] - cost[HYBRID])
    note(f"💰 <b>Step 5: the bottom line.</b> Our AI costs <b>${cost[HYBRID]:,.0f} a month</b>: "
         f"<b>${cost[REACTIVE] - cost[HYBRID]:,.0f} less</b> than normal autoscaling and "
         f"<b>${cost[STATIC] - cost[HYBRID]:,.0f} less</b> than fixed servers. Compared with normal "
         f"autoscaling, that's about <b>${yearly:,.0f} saved per year</b> for this one website.")
    l, r = st.columns(2)
    labels = [f"{ICON[p]} {NAME[p]}" for p in ORDER]
    with l:
        fig = go.Figure(go.Bar(x=[cost[p] for p in ORDER], y=labels, orientation="h",
                               marker_color=[COLORS[p] for p in ORDER],
                               text=[f"${cost[p]:,.0f}" for p in ORDER], textposition="outside"))
        fig.update_layout(title="Monthly server bill (shorter = cheaper)", height=360,
                          xaxis=dict(range=[0, 1.2 * cost.max()], showticklabels=False),
                          yaxis=dict(autorange="reversed"), margin=dict(t=50, b=10, l=10, r=10),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        chart(fig)
    with r:
        fig = go.Figure(go.Bar(x=[breach[p] for p in ORDER], y=labels, orientation="h",
                               marker_color=[COLORS[p] for p in ORDER],
                               text=[f"{breach[p]:.1f}% of the time" for p in ORDER], textposition="outside"))
        fig.update_layout(title="How often the website was slow (shorter = better)", height=360,
                          xaxis=dict(range=[0, 1.35 * max(breach.max(), 0.1)], showticklabels=False),
                          yaxis=dict(autorange="reversed"), margin=dict(t=50, b=10, l=10, r=10),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        chart(fig)
    st.markdown('<div class="read">📖 <b>The trade-off:</b> fixed servers are almost never slow, but they cost the '
                'most because most servers sit idle. Normal autoscaling is cheaper but slow during every rush. '
                'Our AI gets <b>the lowest bill</b> while keeping the site fast most of the time.</div>',
                unsafe_allow_html=True)
    st.write("")
    table = pd.DataFrame({
        "Method": labels,
        "Monthly bill": [f"${cost[p]:,.0f}" for p in ORDER],
        "Website slow": [f"{breach[p]:.1f}% of the time" for p in ORDER],
        "Servers kept busy": [f"{S.loc[p, 'Avg CPU util (%)']:.0f}%" for p in ORDER],
        "Average servers": [f"{S.loc[p, 'Avg pods']:.1f}" for p in ORDER],
    })
    st.table(table.set_index("Method"))
    page_buttons(5)

# ===========================================================================
# TRY IT YOURSELF
# ===========================================================================
if page == PAGES[6]:
    note("🧪 <b>Try it yourself.</b> What happens if the website gets more (or fewer) visitors? "
         "The AI re-plans in a safe simulation, without touching any real servers.")
    pct = st.slider("Change in visitors (%)", -50, 200, 30, 10)
    with st.spinner("Simulating ..."):
        W = cached_what_if(SCENARIOS[scenario_label], pct)
    (dfn, sn), (dfk, sk) = W["AI plan for new traffic"], W["Keep current plan"]
    dcost = sn["Monthly cost ($)"] - cost[HYBRID]
    fast_ai = sn["p95 response (ms)"] <= C["max_rt"]
    fast_keep = sk["p95 response (ms)"] <= C["max_rt"]
    l, r = st.columns(2)
    l.markdown(f"""<div class="card" style="border-top:7px solid {GOOD}">
      <h4>🚀 With our AI (it re-plans)</h4>
      <div class="big">🖥️ Servers: <b>{S.loc[HYBRID, 'Avg pods']:.0f} → {sn['Avg pods']:.0f}</b> on average<br>
      ⚡ Website: <b>{'fast ✅' if fast_ai else 'slow ❌'}</b>, slow only {sn['SLA breaches (%)']:.1f}% of the time<br>
      💰 Bill: <b>{'+' if dcost >= 0 else '-'}${abs(dcost):,.0f} per month</b></div></div>""",
               unsafe_allow_html=True)
    r.markdown(f"""<div class="card" style="border-top:7px solid {BAD}">
      <h4>🙈 If we change nothing</h4>
      <div class="big">🖥️ Servers: same as today<br>
      ⚡ Website: <b>{'fast ✅' if fast_keep else 'slow ❌'}</b>, slow {sk['SLA breaches (%)']:.1f}% of the time<br>
      💰 Bill: about the same, but {'customers suffer' if sk['SLA breaches (%)'] > sn['SLA breaches (%)'] + 1
                                     else 'servers sit idle' if pct < 0 else 'no big difference'}</div></div>""",
               unsafe_allow_html=True)
    st.write("")
    day = st.select_slider("Day to show", days_lbl, value=default_day, key=f"wi_day_{scenario_label}")
    mk = day_mask(day)
    fig = go.Figure()
    fig.add_scatter(x=tt[mk], y=dfn["cpu"][mk], fill="tozeroy", fillcolor="rgba(107,114,128,.25)",
                    line=dict(color=INK, width=1), name=f"Work needed ({pct:+d}% visitors)")
    fig.add_scatter(x=tt[mk], y=dfk["capacity"][mk], name="🙈 Capacity if we change nothing",
                    line=dict(color=BAD, width=2.5, shape="hv", dash="dot"))
    fig.add_scatter(x=tt[mk], y=dfn["capacity"][mk], name="🚀 Capacity with our AI",
                    line=dict(color=GOOD, width=3, shape="hv"))
    fig.update_layout(height=400, yaxis_title="CPUs", **PLOT)
    chart(fig)
    page_buttons(6)

# ===========================================================================
# REAL-WORLD SETUP
# ===========================================================================
if page == PAGES[7]:
    note("🏗️ <b>How this would run in a real company.</b> This demo uses a <b>simulator</b> so it is free and works "
         "offline. In production, each box below is a real cloud service.")
    c = st.columns([4, 1, 4, 1, 4, 1, 4, 1, 4])
    for col, (i, t, d) in zip(c[::2], [
        ("👥", "Users", "Visit the website"),
        ("☸️", "Kubernetes (AWS EKS)", "Runs the website on servers (pods)"),
        ("📊", "Prometheus", "Collects CPU, memory and visitors every 5 min"),
        ("🤖", "AI engine", "Predicts, spots problems, decides"),
        ("⚙️", "Autoscaler", "Adds or removes pods"),
    ]):
        col.markdown(f'<div class="step"><div class="i">{i}</div><div class="t">{t}</div>'
                     f'<div class="d">{d}</div></div>', unsafe_allow_html=True)
    for col in c[1::2]:
        col.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)
    st.caption("🔁 A feedback loop: after each change, the new metrics go back into the AI, so it keeps learning.")
    st.write("")
    st.markdown("""
| In this demo | In a real company |
|---|---|
| Simulated website traffic | Real users through a load balancer |
| Simulated Kubernetes cluster | Kubernetes / AWS EKS |
| Metrics kept in memory | Prometheus + AWS CloudWatch |
| AI engine in Python | Same AI, running as a FastAPI service |
| Simulator adds/removes pods | `kubectl scale`, Kubernetes HPA, AWS Auto Scaling |
| This dashboard | Grafana or a React web app |
""")
    st.subheader("📚 Words used in this project")
    g = st.columns(3)
    for i, (w, m) in enumerate([
        ("Pod / server", "One copy of the website running on a machine. More pods = can handle more visitors."),
        ("Autoscaling", "Automatically adding or removing pods as traffic changes."),
        ("SLA", "The promise to users, e.g. every page loads in under 0.5 s. Breaking it = a 'slow moment'."),
        ("Forecasting", "Predicting future values from past data. Here: Gradient-Boosted Trees (ML)."),
        ("Reinforcement Learning", "The AI learns by trial and error which choice gives the lowest cost + fewest slow moments."),
        ("Anomaly detection", "Spotting behaviour that is far from normal (Robust Covariance model)."),
    ]):
        g[i % 3].markdown(f'<div class="card" style="margin-bottom:12px"><b>{w}</b><br>'
                          f'<span style="opacity:.85">{m}</span></div>', unsafe_allow_html=True)
    page_buttons(7)
