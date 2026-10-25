# ☁️ AI-Driven Intelligent Cloud Resource Allocation and Predictive Autoscaling

An AI that decides **how many servers a website needs**. It adds servers **before** a rush of visitors,
removes them when it is quiet, and explains every decision.

## The problem

A website runs on servers in the cloud, and you **pay for every server** that is running.

* **Too many servers** → money is wasted on idle machines.
* **Too few servers** → the website becomes **slow** when many people visit.

The usual fix, **normal autoscaling** (Kubernetes HPA), adds servers only **after** they are already overloaded.
A new server takes minutes to start, so users suffer in the meantime.

> **Simple example:** a restaurant that calls extra waiters only after the queue is already long.
> Our AI looks at past patterns and calls them **before** the dinner rush.

## What the AI does

| Step | What it does | How (technical name) |
|---|---|---|
| 👀 **Watch** | Reads CPU, memory and number of visitors every 5 minutes | Monitoring metrics |
| 🔮 **Predict** | Guesses how busy the website will be in the next 15 minutes, plus a safety margin | Gradient-Boosted Trees + conformal prediction interval |
| ⚙️ **Scale early** | Adds servers before the rush, removes them when it is quiet | Predictive autoscaling |
| 🧠 **Learn** | Learns by trial and error how much spare capacity is worth paying for | Reinforcement Learning (Q-learning) |
| ⚠️ **Spot problems** | Alerts on sudden traffic jumps, runaway CPU or memory leaks | Anomaly detection (Robust Covariance / MCD) |
| 💬 **Explain** | Writes down why it made each decision, e.g. "Added 5 servers because predicted CPU is 96%" | Explainable AI |
| 🧪 **What-if** | Tests "what if visitors grow 30%?" safely, without touching real servers | Simulation |
| 💰 **Save money** | Shows the current bill vs the AI's bill, per month and per year | Cost optimisation |

## Results

Tested on 7 days of data the AI had **never seen** (normal online shop scenario):

| Method | Monthly bill | Website slow | Servers kept busy |
|---|---|---|---|
| 🧱 Fixed servers (always enough for the busiest moment) | $5,328 | 0.4% of the time | 34% |
| 🐢 Normal autoscaling (Kubernetes-style) | $3,263 | 4.3% of the time | 56% |
| 🔮 AI prediction only | $2,975 | 1.9% of the time | 61% |
| 🚀 **Our AI (prediction + learning)** | **$2,906** | 2.3% of the time | 62% |

**In short:** our AI is about **45% cheaper than fixed servers** and about **11% cheaper than normal autoscaling**,
with roughly **half the slow moments** of normal autoscaling.
Problem detection found **26 of 27** hidden problems across 9 test runs, and named the right type each time.

## How to run it

**Easiest (Windows):** double-click **`Run Dashboard v3.bat`**. The browser opens at http://localhost:8503.
The first run trains the AI (about 25 seconds); after that it is instant.

| Launcher | Dashboard | Address |
|---|---|---|
| `Run Dashboard v3.bat` | ✅ **Easy dashboard**: guided pages, Next/Back buttons, dark mode | http://localhost:8503 |
| `Run Dashboard v2.bat` | Earlier version of the easy dashboard | http://localhost:8502 |
| `Run Dashboard.bat` | Full technical dashboard with all settings | http://localhost:8501 |

**By hand:**
```bash
pip install -r requirements.txt
streamlit run app_v3.py
```

## Tools used

| Tool | What we used it for |
|---|---|
| **Python** | The whole project is written in it |
| **Streamlit** | Turns the Python code into the dashboard website |
| **scikit-learn** | The prediction model and the problem-detection model |
| **NumPy + Pandas** | Maths and data tables |
| **Plotly** | All the interactive charts |
| **Docker** | Packs the app into a container that runs the same on any computer |
| **Kubernetes** | Deployment files to run the app in a cluster |
| **Git + GitHub** | Saving and sharing the code |

## How we tested it (in simple words)

**1. We split 4 weeks of data into 3 parts, so the test is fair.**

| Days | What happens |
|---|---|
| Days 1–14 | The AI **learns** the traffic patterns (training) |
| Days 15–21 | The AI **practises**: it sets its safety margin and the learning agent (RL) learns by trial and error |
| Days 22–28 | The **final test**. The AI has never seen these days, so the results are honest |

**2. The simulated servers behave like real ones.**
* A new server (pod) takes a few minutes to start, and we pay for it even while it is starting.
* The busier the servers are, the slower the website gets. When servers are almost 100% busy,
  response time shoots up (formula: `response time = 60 ms / (1 − how busy the servers are)`).
* A **"slow moment"** (SLA breach) is counted when the website drops work, takes longer than the
  time limit to respond, or runs out of memory.

**3. We compared our AI with two common methods.**
* **Fixed servers:** always run enough servers for the busiest moment ever seen. Never slow, but very costly.
* **Normal autoscaling:** add 2 servers when they get too busy, remove 1 when they are under 40% busy,
  and wait 15 minutes between changes. This is how a basic Kubernetes autoscaler works.

**4. You can test it on your own data** (full dashboard, `app.py`).
Upload a CSV file with a `cpu` column: one row for every 5 minutes, at least 8 days of data.
The `requests` (visitors) and `memory` columns are optional.

## Project files

```
app_v3.py            Easy dashboard (the one to demo): guided pages, Next/Back buttons, dark mode
app_v2.py            Earlier version of the easy dashboard
app.py               Full technical dashboard with all settings
core/simulator.py    Creates website traffic and pretends to be a Kubernetes cluster (servers, start-up delay, cost)
core/ai_models.py    The AI: prediction model, safety margin, learning agent (Q-learning)
core/anomaly.py      Spots unusual behaviour (problem detection) and checks it against hidden test problems
core/pipeline.py     Runs everything end to end, writes the explanations, runs what-if tests
Dockerfile           Recipe to build the Docker container; docker-compose.yml starts it
k8s/                 Kubernetes files (Deployment, Service, autoscaler)
```

## Run with Docker

```bash
docker compose up --build
```
Then open http://localhost:8501. It runs the easy dashboard (`app_v3.py`) by default. To run the full technical
dashboard instead, change `APP_FILE` to `app.py` in `docker-compose.yml`.

## Run on Kubernetes

The `k8s/` folder has 4 files:

| File | What it does, simply |
|---|---|
| `namespace.yaml` | Gives the app its own "room" in the cluster (`ai-cloud-allocator`) |
| `deployment.yaml` | Runs the dashboard as a pod, with CPU/memory limits and health checks that restart it if it hangs |
| `service.yaml` | Opens the app on port 8501 so a browser can reach it, and keeps each user on the same pod |
| `hpa.yaml` | Normal Kubernetes autoscaling (1 to 5 pods when CPU is above 70%), the same method our AI is compared with |

On Docker Desktop with Kubernetes switched on:
```bash
docker build -t ai-cloud-allocator:latest .
kubectl apply -f k8s/
kubectl get pods -n ai-cloud-allocator
```
Then open http://localhost:8501. The autoscaler (`hpa.yaml`) needs `metrics-server` installed in the cluster.

## Demo vs real company

This project uses a **simulator** instead of real cloud servers, so it is free and works offline.
In a real company, each part would be replaced like this:

| In this project | In a real company |
|---|---|
| Simulated Kubernetes cluster | Kubernetes / AWS EKS |
| Simulated metrics | Prometheus + AWS CloudWatch |
| AI adds/removes servers in the simulator | Kubernetes HPA / KEDA / `kubectl scale` / AWS Auto Scaling |
| Streamlit dashboard | Grafana, or a React website on top of a FastAPI service |
