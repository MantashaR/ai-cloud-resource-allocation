# ☁️ CloudMind: AI-Driven Intelligent Cloud Resource Allocation and Predictive Autoscaling

A working prototype that allocates cloud resources (Kubernetes pods) using AI:

1. **Predicts** future CPU, memory and request rate (Gradient-Boosted Trees + conformal prediction interval)
2. **Scales before the spike** (predictive autoscaling) instead of waiting for high CPU
3. **Learns the cost vs SLA trade-off** with Reinforcement Learning (Q-learning)
4. **Detects anomalies** such as traffic spikes, runaway CPU and memory leaks (Robust Covariance / MCD)
5. **Explains every decision** ("Scale Up 14 → 19 because predicted CPU is 96%, SLA at risk ...")
6. **What-if simulation** ("what if traffic grows 30%?") without touching the live cluster
7. **Cost optimization**: current vs recommended resources, monthly and yearly savings

## Run it

**Windows:** double-click **`Run CloudMind.bat`**. The browser opens at http://localhost:8501.
The first run trains the AI (~25 s); after that the results are cached.

Manual:
```bash
pip install -r requirements.txt
streamlit run app.py
```

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
app.py               Streamlit dashboard (8 tabs)
core/simulator.py    Workload generator, Kubernetes-style cluster simulator, latency model, baseline autoscalers
core/ai_models.py    Forecaster, conformal calibration, predictive policy, Q-learning agent, HPA stabilizer
core/anomaly.py      Robust-covariance anomaly detector + validation against injected anomalies
core/pipeline.py     End-to-end experiment, explainable decisions, what-if engine
```

## Method

* **Data split:** train forecasters on days 1-14 → calibrate the interval and train RL on days 15-21 →
  test everything on days 22-28, which the models never saw.
* **Cluster model:** pods take a configurable start-up delay (node + pod start) and are billed while starting.
  Response time follows a queueing model, `RT = 60 ms / (1 - utilization)`.
  An SLA breach is dropped work, response time above the limit, or out-of-memory.
* **Baselines:** static peak provisioning, and a reactive threshold autoscaler (+2 pods above the max
  utilization, −1 pod below 40%, 15-minute cooldown).
* **Your own data:** upload a CSV with a `cpu` column (vCPU per 5 minutes, at least 8 days). `requests` and
  `memory` columns are optional.

## Production mapping

| Prototype | Production |
|---|---|
| Cluster simulator | Kubernetes / AWS EKS |
| Metrics series | Prometheus + CloudWatch |
| Decision engine output | HPA / KEDA / `kubectl scale` / EC2 Auto Scaling |
| Streamlit dashboard | Grafana / React frontend over a FastAPI service |
