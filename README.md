# 🚨 Multimodal Customer Risk & Escalation Engine

> An end-to-end production ML system that predicts customer support ticket escalation risk by combining structured data analysis with natural language understanding.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-orange)](https://lightgbm.readthedocs.io)
[![DistilBERT](https://img.shields.io/badge/DistilBERT-HuggingFace-yellow)](https://huggingface.co/distilbert-base-uncased)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-purple)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Versioned-green)](https://dvc.org)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running The Project](#running-the-project)
- [API Documentation](#api-documentation)
- [Data Drift Detection](#data-drift-detection)
- [Docker Deployment](#docker-deployment)
- [Interview Highlights](#interview-highlights)

---

## 🎯 Project Overview

This project builds a **production-grade, multimodal ML system** that automatically identifies high-risk customer support tickets that are likely to escalate — before they do.

### The Business Problem

In enterprise support environments, thousands of tickets arrive daily. Support managers cannot manually review every ticket. Missing a genuinely frustrated customer leads to:

- Customer churn
- Negative public reviews
- Escalation to senior management
- SLA penalties

### The Solution

An automated risk scoring engine that:

1. **Reads** ticket metadata (priority, resolution time, CSAT score)
2. **Understands** the actual language the customer used (NLP)
3. **Combines both signals** using Late Fusion
4. **Outputs** a risk score with human-readable explanation
5. **Explains why** a ticket was flagged (SHAP)

### Real World Impact

```
Without system → agent manually reviews 1000 tickets/day
With system    → agent reviews only top 100 flagged tickets
                 catches 84% of real escalations
                 reduces review workload by 90%
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
│                                                              │
│  PostgreSQL → DataIngestion → DataValidation                │
│           → DataTransformation → ModelTrainer               │
│           → NLPTrainer → LateFusionTrainer                  │
│                                                              │
│  Outputs: fusion_model.pkl, scaler.pkl, label_encoders.pkl  │
└──────────────────────┬──────────────────────────────────────┘
                       │ artifacts
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     SERVING LAYER                            │
│                                                              │
│  New Ticket → PredictPipeline                               │
│            → transform_input() [same transforms as training]│
│            → extract_text_embedding() [DistilBERT]          │
│            → concatenate [12 tabular + 768 NLP = 780 feats] │
│            → fusion_model.predict_proba()                   │
│            → SHAP explanation                               │
│            → PredictionResult                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐      ┌──────────────────────┐
│   FastAPI       │      │   Streamlit           │
│   /predict      │      │   Dashboard           │
│   /health       │      │   Risk Score Display  │
│   /model-info   │      │   SHAP Visualization  │
│   /docs         │      │   Ticket Input Form   │
└─────────────────┘      └──────────────────────┘
```

---

## 🔀 Multimodal Late Fusion

```
Tabular Path                    NLP Path
────────────                    ────────
12 structured features          combined_text
resolution_time_hours     →     initial_message +
csat_score                      agent_first_reply +
priority                        resolution_summary
sla_plan                              │
customer_segment                      ▼
...                             DistilBERT
      │                         distilbert-base-uncased
      ▼                               │
 LightGBM / XGBoost                   ▼
 tabular signals               768-dim CLS embedding
      │                               │
      └──────────────┬────────────────┘
                     ▼
            np.concatenate → (1, 780)
                     │
              LightGBM Fusion Model
                     │
              Risk Score + SHAP
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Data** | PostgreSQL + pgAdmin4 | Raw data storage |
| **Ingestion** | SQLAlchemy + psycopg2 | Database connection |
| **Versioning** | DVC | Data and model versioning |
| **Validation** | Evidently AI 0.4.16 | Data drift detection |
| **Tabular ML** | XGBoost, LightGBM, sklearn | Tabular model training |
| **NLP** | DistilBERT (HuggingFace) | Text embedding extraction |
| **Explainability** | SHAP TreeExplainer | Prediction explanations |
| **Tracking** | MLflow | Experiment tracking |
| **Serving** | FastAPI + Uvicorn | REST API |
| **Dashboard** | Streamlit + Plotly | Business UI |
| **Deployment** | Docker + Render | Containerization |

---

## 📊 Dataset

- **Source:** Synthetic IT Support Tickets (Kaggle)
- **Size:** 100,000 tickets × 20 columns
- **Date Range:** 2024 (with temporal drift simulation)
- **Storage:** PostgreSQL database

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `initial_message` | Text | Customer's complaint |
| `agent_first_reply` | Text | Agent's response |
| `resolution_summary` | Text | How ticket was resolved |
| `resolution_time_hours` | Float | Time to resolve |
| `customer_sentiment` | Categorical | Sentiment label |
| `csat_score` | Integer | Satisfaction score 0-5 |
| `status` | Categorical | Ticket status |
| `priority` | Categorical | low/medium/high/urgent |

### Target Variable Engineering

No ground-truth escalation label existed in the dataset. Target was engineered using **AND-logic proxy labeling**:

```python
escalated = (
    customer_sentiment isin ['negative', 'very_negative']  AND
    csat_score <= 2                                         AND
    status isin ['closed_no_action', 'open', 'on_hold']
)
```

**Why AND logic:**
- OR logic caused data leakage → AUC 1.0 (each condition directly defined target)
- Single condition (reopened) had no learnable signal → AUC 0.50
- AND logic requires all three signals simultaneously → realistic 10% escalation rate → AUC 0.84

---

## 🤖 ML Pipeline

### Training Pipeline

```
main.py
   │
   ├── DataIngestion        → pulls from PostgreSQL via SQLAlchemy
   ├── DataValidation       → schema check + Evidently drift detection
   ├── DataTransformation   → FE, encoding, scaling, train/test split
   ├── ModelTrainer         → trains 5 models, MLflow tracking
   ├── NLPTrainer           → DistilBERT CLS embeddings (GPU)
   └── LateFusionTrainer    → concatenate + final LightGBM
```

### Models Trained

| Model | Purpose |
|-------|---------|
| Decision Tree | Interpretable baseline |
| Random Forest | Parallel ensemble comparison |
| XGBoost Baseline | Sequential boosting baseline |
| XGBoost Tuned | Tuned with subsample, colsample |
| LightGBM | Fast boosting — final tabular model |
| **LateFusion LightGBM** | **Final production model** |

### Class Imbalance Handling

```python
# 10% escalation rate → significant imbalance
scale_pos_weight = neg / pos    # ~9x for XGBoost
class_weight = 'balanced'       # for sklearn models
```

### Missing Value Strategy

```python
# Meaningful missingness — never mode-fill
resolution_time_hours → fill with -1 (sentinel) + is_unresolved flag
resolution_summary    → fill with "unresolved"
region                → fill with "Unknown"
```

---

## 📈 Model Performance

### Tabular Models Comparison

| Model | Accuracy | F1 Score | Precision | Recall | ROC AUC |
|-------|----------|----------|-----------|--------|---------|
| Decision Tree | 0.7196 | 0.3700 | 0.2392 | 0.8167 | 0.8341 |
| Random Forest | 0.7260 | 0.3684 | 0.2400 | 0.7925 | 0.8293 |
| XGBoost Baseline | 0.7184 | 0.3734 | 0.2407 | 0.8318 | 0.8395 |
| XGBoost Tuned | 0.7135 | 0.3722 | 0.2389 | **0.8421** | **0.8411** |
| **LightGBM** | 0.7141 | **0.3727** | 0.2393 | **0.8421** | 0.8407 |

### Why Recall Over Accuracy

```
Missing a real escalation  → customer churns, bad reviews
False alarm                → agent spends 5 extra minutes

Cost of false negative >> Cost of false positive
Therefore: optimize Recall, accept lower Precision
```

### Confusion Matrix (Best Model)

```
                 Predicted
              Not Esc  Escalated
Actual Not Esc  15730     6749    ← false alarms (acceptable)
       Escalated  398     2123    ← caught 84% of real escalations
```

---

## 📁 Project Structure

```
Customer-Risk-Escalation-Engine/
│
├── 📄 main.py                         # Training pipeline entry point
├── 📄 api.py                          # FastAPI prediction server
├── 📄 app.py                          # Streamlit business dashboard
├── 📄 Dockerfile                      # Container recipe
├── 📄 docker-compose.yml              # Multi-service orchestration
├── 📄 requirements.txt                # Python dependencies
├── 📄 render.yaml                     # Render deployment config
├── 📄 schema.yaml                     # Data validation schema
│
├── 📁 Customer_Risk_Escalation/
│   ├── 📁 components/
│   │   ├── data_ingestion.py          # PostgreSQL → raw DataFrame
│   │   ├── data_validation.py         # Schema + drift detection
│   │   ├── data_transformation.py     # Feature engineering pipeline
│   │   ├── model_trainer.py           # Train 5 models + MLflow
│   │   ├── nlp_trainer.py             # DistilBERT embeddings
│   │   └── late_fusion_trainer.py     # Multimodal fusion model
│   │
│   ├── 📁 pipeline/
│   │   ├── train_pipeline.py          # Orchestrates training
│   │   └── predict_pipeline.py        # Serves predictions + SHAP
│   │
│   ├── 📁 entity/
│   │   ├── config_entity.py           # Typed configs per component
│   │   └── artifact_entity.py         # Typed outputs per component
│   │
│   ├── 📁 database/
│   │   └── sql_download.py            # SQLClient utility
│   │
│   ├── 📁 exceptions/
│   │   └── exception.py               # Custom exception handler
│   │
│   ├── 📁 logger/
│   │   └── logging.py                 # Custom logger
│   │
│   └── 📁 constant/
│       └── __init__.py                # All project constants
│
├── 📁 artifacts/                      # Generated by pipeline
│   └── timestamp/
│       ├── data_ingestion/
│       ├── data_validation/
│       ├── data_transformation/
│       ├── model_trainer/
│       ├── nlp_trainer/
│       └── late_fusion/
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_nlp_distilbert.ipynb
│   └── 05_late_fusion.ipynb
│
└── 📁 data/
    ├── raw_data/
    └── processed_data/
```

---

## ⚙️ Setup & Installation

### Prerequisites

```
Python 3.11+
PostgreSQL 14+
Git
NVIDIA GPU (optional — CPU fallback available)
```

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Risk-Escalation-Engine.git
cd Customer-Risk-Escalation-Engine
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup PostgreSQL

```bash
# Create database in pgAdmin4
Database name: customer_escalation

# Import dataset
# Download from Kaggle: Synthetic IT Support Tickets
# Import CSV to table: escalation_dataset
```

### 5. Configure Environment Variables

```bash
# Create .env file in project root
SQL_DB_URL=postgresql://username:password@localhost:5432/customer_escalation
```

---

## 🚀 Running The Project

### Train The Full Pipeline

```bash
python main.py
```

This runs the complete pipeline:
```
DataIngestion → DataValidation → DataTransformation
→ ModelTrainer → NLPTrainer → LateFusionTrainer
```

Training time estimate:
```
Tabular models    → ~5 minutes
DistilBERT (GPU)  → ~10 minutes for 80K tickets
DistilBERT (CPU)  → ~60-90 minutes
Late Fusion       → ~3 minutes
```

### Start FastAPI Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: `http://localhost:8000/docs`

### Start Streamlit Dashboard

```bash
# New terminal
streamlit run app.py
```

Dashboard: `http://localhost:8501`

---

## 📡 API Documentation

### Base URL
```
Local:      http://localhost:8000
Production: https://your-app.onrender.com
```

### Endpoints

#### `GET /health`
```json
{
  "status": "running",
  "pipeline_loaded": true,
  "model": "LateFusion_LightGBM"
}
```

#### `GET /model-info`
```json
{
  "model_name": "Late Fusion LightGBM",
  "tabular_features": 12,
  "embedding_dim": 768,
  "total_features": 780,
  "base_model": "distilbert-base-uncased"
}
```

#### `POST /predict`

**Request:**
```json
{
  "customer_segment": "individual",
  "product_area": "billing",
  "issue_type": "account_access",
  "priority": "urgent",
  "status": "on_hold",
  "sla_plan": "standard",
  "initial_message": "I have been waiting three months...",
  "agent_first_reply": "We will look into this.",
  "resolution_summary": null,
  "resolution_time_hours": null,
  "reopened": 1,
  "customer_sentiment": "very_negative",
  "csat_score": 1,
  "has_attachment": 0,
  "platform": "web",
  "region": "EU",
  "created_at": "2026-07-15T10:30:00"
}
```

**Response:**
```json
{
  "risk_score": 0.8423,
  "risk_level": "Critical",
  "escalated": true,
  "top_reasons": [
    {
      "feature": "resolution_time_hours",
      "impact": "+1.053",
      "direction": "increases risk"
    },
    {
      "feature": "csat_score",
      "impact": "+0.821",
      "direction": "increases risk"
    },
    {
      "feature": "is_unresolved",
      "impact": "+0.634",
      "direction": "increases risk"
    }
  ],
  "model_used": "LateFusion_LightGBM"
}
```

---

## 🔍 Data Drift Detection

Evidently AI monitors feature distributions between reference data (training) and current production data.

```python
# Drift detection in DataValidation component
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=current_df)
```

**Drift behavior:**
```
Schema failure    → pipeline STOPS immediately
Quality failure   → pipeline STOPS immediately
Drift detected    → WARNING logged, pipeline CONTINUES
No drift          → pipeline CONTINUES normally
```

HTML drift report saved to: `artifacts/timestamp/data_validation/reports/`

---

## 🐳 Docker Deployment

### Local Docker

```bash
# Build and start both services
docker-compose up --build

# Run in background
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Render Deployment

1. Push code to GitHub (includes Dockerfile and render.yaml)
2. Connect GitHub repo to Render
3. Render builds Docker image on their servers
4. Set environment variables in Render dashboard:
   - `SQL_DB_URL` → your PostgreSQL connection string
5. Deploy

Services will be available at:
```
API:       https://escalation-api.onrender.com
Dashboard: https://escalation-dashboard.onrender.com
```

---

## 💡 Interview Highlights

### Key Technical Decisions

**1. Why Late Fusion over Early Fusion?**
> Text embeddings and tabular features exist in completely different mathematical spaces. Fusing at the decision layer rather than the input layer gives each model cleaner, more separable signals to learn from.

**2. Why AND-logic for target engineering?**
> OR-logic caused direct leakage — each condition alone fully predicted the target → AUC 1.0. AND-logic requires all three signals simultaneously, creating a genuine learning problem with realistic 10% escalation rate → AUC 0.84.

**3. Why DistilBERT over TF-IDF?**
> TF-IDF treats "I am not happy" and "I am happy" as similar. DistilBERT understands negation and context. For detecting customer frustration in support tickets, semantic understanding matters.

**4. Why Recall over Accuracy?**
> Missing a genuine escalation costs far more than a false alarm. A missed escalation can lead to customer churn, negative reviews, and SLA penalties. A false alarm costs an agent 5 minutes.

**5. How was class imbalance handled?**
> Used `scale_pos_weight = neg/pos ≈ 9` for XGBoost and `class_weight='balanced'` for sklearn models. This tells models to treat each escalation as 9 non-escalations during training.

**6. Why config-artifact pattern?**
> Each pipeline component receives a typed config and returns a typed artifact. This makes every stage independently testable, the data flow explicit and traceable, and the entire pipeline runnable with one command.

**7. Why save LabelEncoder per column?**
> A shared encoder gets overwritten on each fit_transform call — only the last column's mapping is preserved. Separate encoders per column guarantee consistent mapping between training and production inference.

---

## 📊 MLflow Experiment Tracking

```bash
# View all experiments
mlflow ui

# Open browser at
http://127.0.0.1:5000
```

Experiments tracked:
- `customer_escalation_tabular` — 5 model runs with full metrics
- `customer_escalation_nlp` — DistilBERT embedding extraction
- `customer_escalation_fusion` — Late fusion model

---

## 🔧 Troubleshooting

**CUDA not available:**
```bash
# Verify PyTorch CUDA version matches driver
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # check CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**MLflow runs not visible:**
```bash
# Always run from project root
cd Customer-Risk-Escalation-Engine
mlflow ui
```

**Pipeline not loaded in FastAPI:**
```bash
# Always run from project root
uvicorn api:app --host 0.0.0.0 --port 8000
# Not from app/ subfolder
```

**Artifacts not found:**
```bash
# Run training pipeline first
python main.py
# Artifacts generated in artifacts/timestamp/
```

---

## 📄 License

MIT License — free to use for learning and portfolio purposes.

---

## 👤 Author

**FAIZAN RIAZ**  
Aspiring ML Engineer  
[GitHub](https://github.com/faizan23804) | [LinkedIn](https://linkedin.com/in/faizanriaz23)

---

*Built as a portfolio project demonstrating end-to-end ML engineering — from data ingestion through PostgreSQL, modular pipeline architecture, multimodal modeling with DistilBERT + LightGBM late fusion, to production deployment with FastAPI, Streamlit, Docker, and Render.*

