# 🛡️ Anomaly Detection in Online Transactions (Fraud Detection)

## 📌 Overview
This project focuses on **building a robust, modular, and production-ready fraud detection system** for online financial transactions.  
It goes beyond a simple machine learning model — showcasing **MLOps best practices**, **scalable architecture**, and **real-world deployment readiness**.  

The primary goals of this project are:
- Demonstrate **ML skills** (classical ML, anomaly detection, hybrid models).  
- Demonstrate **MLOps skills** (data versioning, experiment tracking, CI/CD, monitoring, API deployment).  
- Build the project **config-driven and modular**, so switching datasets/models requires minimal code changes.  
- Deliver a project that is practical, and production-aligned.

---

## 🎯 Key Features
- **Dataset Flexibility** → Start with the Credit Card Fraud dataset, later switch to IEEE-CIS Fraud dataset (or others) with minimal config changes.  
- **Config-Driven Design** → All dataset, model, and API settings live in YAML configs (`src/config/`).  
- **Modular Pipeline** → Each stage (data preprocessing, training, serving, monitoring) is isolated for reuse.  
- **Data Versioning (DVC)** → Raw and processed data tracked with DVC for reproducibility.  
- **Experiment Tracking (MLflow)** → Track experiments, parameters, metrics, and artifacts.  
- **Baseline → Anomaly Detection → Hybrid Models** → Stepwise model development for clarity and benchmarking.  
- **FastAPI Real-Time Serving** → Deploy fraud detection API with real-time inference.  
- **Monitoring & Drift Detection** → Basic monitoring + drift detection implemented for deployed models.  
- **CI/CD with GitHub Actions** → Automate testing, linting, and deployment workflows.  

---

## 🏗️ Architecture
The project is structured to mimic real-world ML systems with modular components.

fraud-detection-mlops/
│
├── data/                          # Raw + processed data (tracked by DVC, not Git)
│   ├── raw/                       
│   ├── processed/                 
│   └── dvc.yaml                   
│
├── notebooks/                     # Jupyter notebooks for EDA & prototyping
│
├── src/                           # Core source code
│   ├── preprocessing/             # Data cleaning, feature engineering
│   │   └── preprocess.py
│   │
│   ├── models/                    # Training, evaluation, anomaly detection
│   │   ├── baseline.py            # Logistic Regression, RF, etc.
│   │   ├── anomaly.py             # Isolation Forest / AutoEncoder
│   │   └── hybrid.py              # Combination models
│   │
│   ├── api/                       # Serving layer
│   │   ├── app.py                 # FastAPI entrypoint
│   │   └── utils.py
│   │
│   ├── monitoring/                # Monitoring & drift detection
│   │   ├── basic_metrics.py
│   │   ├── drift_detection.py
│   │   └── logger.py
│   │
│   └── config/                    # Config-driven design
│       ├── dataset.yaml           # dataset-specific configs
│       ├── model.yaml             # model params
│       └── api.yaml               # API/monitoring configs
│
├── experiments/                   # MLflow artifacts (gitignored)
│
├── tests/                         # Unit + integration tests
│
├── scripts/                       # Utility scripts (for CI/CD, automation)
│   ├── run_training.sh
│   └── run_api.sh
│
├── .github/                       # GitHub Actions workflows
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt               # Python dependencies
├── dvc.lock                       # DVC lock file
├── README.md                      # Project overview
└── mlflow_config.yaml             # MLflow setup


---

## ⚙️ Tech Stack
- **Languages & Libraries**: Python, scikit-learn, pandas, numpy, matplotlib/seaborn  
- **ML Models**: Logistic Regression, Random Forest, Isolation Forest, Autoencoders  
- **Experiment Tracking**: MLflow  
- **Data Versioning**: DVC  
- **Serving**: FastAPI  
- **Monitoring**: Custom metrics + drift detection (data drift, concept drift)  
- **CI/CD**: GitHub Actions  
- **Optional Extensions**: PostgreSQL or other DBs (future work)  

---

## 🚀 Project Workflow

### 1. Data Pipeline
- Store raw and processed data under DVC.
- Config-driven preprocessing (`src/preprocessing/preprocess.py`).
- Ability to switch datasets by editing `dataset.yaml`.

### 2. Modeling
- **Step 1:** Train classical ML baseline models (Logistic Regression, Random Forest).  
- **Step 2:** Implement anomaly detection (Isolation Forest, Autoencoder).  
- **Step 3:** Build hybrid model combining supervised + anomaly detection.  
- Track all runs with **MLflow**.

### 3. Serving
- Expose models as a **FastAPI** service.
- Endpoint `/predict` for transaction fraud detection.
- Configurable thresholds via YAML.

### 4. Monitoring
- Log model metrics on live traffic.
- Detect data drift (e.g., Kolmogorov-Smirnov, PSI).
- Simple alerting mechanism for anomalies.

### 5. CI/CD
- **GitHub Actions** → run unit tests, linting, and DVC/MLflow checks.
- Automated deployment pipeline for API.

---

## 📊 Monitoring & Drift Detection
- **Basic Metrics**: Accuracy, Precision, Recall, F1, AUC.  
- **Drift Detection**: PSI (Population Stability Index), KS-Test for data drift.  
- **Visualization**: Monitoring dashboards (basic plots/logs).  

---

## 📌 Future Improvements
- Switch storage backend from **DVC** to **PostgreSQL** (DB integration).  
- Deploy with **Docker + Kubernetes** for scalable production.  
- Integrate **real-time streaming data** (Kafka or Spark Streaming).  
- Add **advanced monitoring** (Prometheus + Grafana dashboards).  

---

## 🧑‍🎓 Why This Project Matters
This project is not a “toy fraud detection model.”  
It’s designed to reflect **real-world industry workflows**:
- Reproducibility via **DVC + MLflow**  
- Deployment readiness via **FastAPI + CI/CD**  
- Model robustness via **baseline → anomaly detection → hybrid**  
- Resume-friendly skills: **MLOps, ML, Monitoring, Config-driven design**  

It is a **capstone-style project** that balances academic rigor with professional depth.  

---

## 📂 Setup & Run (Quickstart)

```bash
# Clone the repo
git clone <private-repo-url>
cd fraud-detection-mlops

# Install dependencies
pip install -r requirements.txt

# Run preprocessing + training (tracked via DVC)
dvc repro

# Serve API
uvicorn src.api.app:app --reload

# Run monitoring
python src/monitoring/basic_metrics.py
```
---

📜 License

This project is developed for educational and research purposes.
Commercial usage requires permission.
