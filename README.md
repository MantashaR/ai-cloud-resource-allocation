# ☁️ AI-Driven Intelligent Cloud Resource Allocation and Predictive Autoscaling

A working prototype that allocates cloud resources (Kubernetes pods) using AI:

1. **Predicts** future CPU, memory and request rate (Gradient-Boosted Trees + conformal prediction interval)
2. **Scales before the spike** (predictive autoscaling) instead of waiting for high CPU
3. **Learns the cost vs SLA trade-off** with Reinforcement Learning (Q-learning)
4. **Detects anomalies** such as traffic spikes, runaway CPU and memory leaks (Robust Covariance / MCD)
5. **Explains every decision** ("Scale Up 14 → 19 because predicted CPU is 96%, SLA at risk ...")
6. **What-if simulation** ("what if traffic grows 30%?") without touching the live cluster
7. **Cost optimization**: current vs recommended resources, monthly and yearly savings

## Run it

**Windows:** double-click **`Run Dashboard.bat`**. The browser opens at http://localhost:8501.
The first run trains the AI (~25 s); after that the results are cached.

Manual:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run with Docker

```bash
docker compose up --build
```
Open http://localhost:8501. To run the full technical dashboard instead of the easy one, set
`APP_FILE: app.py` in `docker-compose.yml`.

## Deploy to Kubernetes

The `k8s/` folder has the manifests:

| File | What it does |
|---|---|
| `namespace.yaml` | Own namespace `ai-cloud-allocator` |
| `deployment.yaml` | Runs the dashboard pod with CPU/memory requests and limits, readiness and liveness probes |
| `service.yaml` | Exposes it on port 8501 (LoadBalancer, sticky sessions for Streamlit) |
| `hpa.yaml` | Standard CPU-based HorizontalPodAutoscaler (1 to 5 pods), the reactive baseline the AI is compared with |

On Docker Desktop with Kubernetes enabled:
```bash
docker build -t ai-cloud-allocator:latest .
kubectl apply -f k8s/
kubectl get pods -n ai-cloud-allocator
```
Then open http://localhost:8501. The HPA needs `metrics-server` installed in the cluster.

## Results (built-in e-commerce scenario, 7 unseen test days)

| Autoscaler | Monthly cost | SLA breaches | Avg CPU utilization |
|---|---|---|---|
| Static (peak provisioning) | $5,328 | 0.4% | 34% |
| Reactive (Kubernetes-HPA style) | $3,263 | 4.3% | 56% |
| AI Forecast | $2,975 | 1.9% | 61% |
| **AI Forecast + RL** | **$2,906** | 2.3% | 62% |

Anomaly detection found 26 of 27 injected anomalies across 9 scenario/seed runs, each with the correct type.

## Project structure

```
app_v3.py            Easy-to-read dashboard: guided pages, Next/Back buttons, dark mode switch
app_v2.py            Earlier version of the easy dashboard
app.py               Full technical dashboard (8 tabs)
Dockerfile           Container image; docker-compose.yml runs it
k8s/                 Kubernetes manifests (Deployment, Service, HPA)
core/simulator.py    Workload generator, Kubernetes-style cluster simulator, latency model, baseline autoscalers
core/ai_models.py    Forecaster, conformal calibration, predictive policy, Q-learning agent, HPA stabilizer
core/anomaly.py      Robust-covariance anomaly detector + validation against injected anomalies
core/pipeline.py     End-to-end experiment, explainable decisions, what-if engine
```

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

## Production mapping

| Prototype | Production |
|---|---|
| Cluster simulator | Kubernetes / AWS EKS |
| Metrics series | Prometheus + CloudWatch |
| Decision engine output | HPA / KEDA / `kubectl scale` / EC2 Auto Scaling |
| Streamlit dashboard | Grafana / React frontend over a FastAPI service |
