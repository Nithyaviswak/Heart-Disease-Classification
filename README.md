# Heart Disease Analytics Platform v2.0

Enterprise healthcare analytics platform with React frontend, Flask REST API, 5 ML/DL models (sklearn + TensorFlow + PyTorch), SHAP explainability, JWT authentication, and Power BI integration.

## Models

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 80.98% | 76.19% | 91.43% | 83.12% | 92.98% |
| **Random Forest** | **100%** | **100%** | **100%** | **100%** | **100%** |
| SVM | 92.68% | 91.67% | 94.29% | 92.96% | 97.71% |
| TensorFlow MLP | 100% | 100% | 100% | 100% | 100% |
| PyTorch MLP | 96.59% | 95.37% | 98.10% | 96.71% | 99.70% |

## Features

- **1025 samples** — cleaned heart disease dataset
- **5 Models** — Logistic Regression, Random Forest, SVM, TensorFlow MLP, PyTorch MLP
- **BatchNorm + Dropout** — DL models use BatchNorm, ReLU, Dropout (0.3) for regularization
- **Early Stopping** — DL models restore best weights on validation loss plateau
- **SHAP Explainability** — every sklearn prediction shows top feature contributions
- **JWT Authentication** — role-based access (admin/viewer) with token refresh
- **React Frontend** — dashboard with KPIs, risk assessment, prediction history, analytics charts
- **SQLite Database** — predictions, users, model metrics, patients — all persisted
- **Power BI Integration** — pre-built datasets, DAX measures, and 6-page dashboard spec
- **Report Generation** — CSV, Excel, and PDF export of clinical data
- **REST API** — 18 endpoints

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Train all 5 models
python backend/ml/train.py

# Build frontend
cd frontend && npm run build && cd ..

# Run production server
python app.py
# -> http://localhost:5000
```

## Development

```bash
# Backend (port 5000)
python app.py

# Frontend (port 3000, proxies /api to :5000)
cd frontend && npm run dev
```

## Default Credentials

- **Username**: admin
- **Password**: admin123

> ⚠️ **Security Warning:** Change default credentials immediately after first login. Never use default credentials in production.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login, get JWT token |
| GET | `/api/auth/me` | Get current user info |
| POST | `/api/predict` | Predict with SHAP explanation |
| POST | `/api/predict/explain` | Predict with all models |
| GET | `/api/history` | Prediction history (paginated) |
| GET | `/api/history/stats` | Prediction statistics |
| GET | `/api/dashboard/kpi` | Executive KPIs |
| GET | `/api/dashboard/summary` | Dashboard summary |
| GET | `/api/analytics/summary` | Full analytics breakdown |
| GET | `/api/model-metrics` | Model performance metrics |
| GET | `/api/models` | List available models |
| GET | `/api/download/csv/<table>` | Export table as CSV |
| GET | `/api/download/excel` | Export full report (Excel) |
| GET | `/api/download/pdf/<report_type>` | Export report (PDF) |
| GET | `/api/health` | Health check |

## Project Structure

```
heart-disease-classification/
├── app.py                        # Production entry point
├── requirements.txt              # Python dependencies
├── data.py                       # Data loading & preprocessing
├── data/
│   └── heart_disease.csv         # 1025-sample heart disease dataset
├── models/                       # Trained ML/DL models
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── tensorflow_mlp.keras      # TensorFlow/Keras MLP
│   ├── pytorch_mlp.pt            # PyTorch MLP
│   ├── model_metrics.json
│   └── scaler.pkl
├── backend/
│   ├── app.py                    # Flask factory (create_app)
│   ├── config.py                 # Configuration
│   ├── database.py               # SQLite schema & helpers
│   ├── auth.py                   # JWT authentication
│   ├── analytics.py              # Analytics engine
│   ├── reports.py                # Report generation (CSV/Excel/PDF)
│   ├── ml/
│   │   ├── models.py             # ModelTrainer, MLService, ShapExplainer
│   │   └── train.py              # Training pipeline (sklearn + TF + PyTorch)
│   └── routes/
│       ├── auth.py               # Auth endpoints
│       ├── predict.py            # Prediction endpoints
│       ├── history.py            # History endpoints
│       ├── dashboard.py          # Dashboard & analytics
│       ├── download.py           # Export endpoints
│       └── metrics.py            # Model metrics
├── frontend/                     # React + Vite + Tailwind
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── index.css
│       ├── App.jsx               # Routes & auth guard
│       ├── lib/
│       │   ├── api.js            # Axios with JWT interceptor
│       │   └── auth.jsx          # AuthContext provider
│       ├── components/
│       │   ├── Layout.jsx        # Sidebar navigation
│       │   └── ui/card.jsx
│       └── pages/
│           ├── Login.jsx
│           ├── Dashboard.jsx
│           ├── Predict.jsx
│           ├── History.jsx
│           ├── Analytics.jsx
│           └── Patients.jsx
├── analytics/
│   └── generate_datasets.py      # Power BI CSV dataset generator
├── powerbi/
│   ├── dashboard_spec.txt        # Dashboard spec with DAX + M code
│   └── *.csv                     # Generated Power BI datasets
├── models/
│   ├── tensorflow_mlp.py         # TensorFlow/Keras MLP (BatchNorm + Dropout)
│   └── pytorch_mlp.py            # PyTorch MLP (BatchNorm + Dropout)
└── reports/                      # Generated PDF/Excel reports
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, Vite 8, Tailwind CSS 4, Recharts |
| Backend | Flask 3, flask-cors, JWT, SQLite |
| ML | scikit-learn, joblib, SHAP |
| Deep Learning | TensorFlow/Keras 3, PyTorch 2.4 |
| Export | ONNX, onnxruntime |
| Deployment | Render, gunicorn |

## Security

A full security audit was conducted on 2026-07-17. Below is a summary of findings and hardening steps.

### Vulnerability Summary

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 Critical | 3 | Identified — SQL injection in reports, auth decorators not applied |
| 🟠 High | 4 | Identified — weak hashing, hardcoded secrets, open CORS, pickle risk |
| 🟡 Medium | 3 | Identified — no input validation, default creds, DB in repo |
| 🔵 Low | 2 | Identified — debug mode, missing security headers |

### Critical Issues

1. **SQL Injection** — `reports.py` and `download.py` interpolate user-supplied table names directly into SQL queries. Must whitelist allowed table names.
2. **Broken Authentication** — `token_required` is imported inside route functions but never applied as a decorator. All "protected" endpoints are publicly accessible.
3. **Auth Decorator Bypass** — Routes in `predict.py`, `history.py`, `download.py`, `dashboard.py`, and `metrics.py` import `token_required` but do not use `@token_required`.

### Production Hardening Checklist

- [ ] Fix SQL injection: whitelist table names in `export_csv()` and `export_excel()`
- [ ] Apply `@token_required` decorator to all protected routes
- [ ] Replace SHA-256 password hashing with `bcrypt` or `argon2`
- [ ] Set strong `SECRET_KEY` and `JWT_SECRET_KEY` via environment variables
- [ ] Restrict CORS origins: `CORS(app, origins=["https://yourdomain.com"])`
- [ ] Add input validation on registration (email format, password strength, role restriction)
- [ ] Add `flask-talisman` for security headers (CSP, HSTS, X-Frame-Options)
- [ ] Remove default credentials or force password change on first login
- [ ] Add `.db`, `.db-shm`, `.db-wal` to `.gitignore`
- [ ] Use `gunicorn` in production (never `app.run(debug=True)`)
- [ ] Enable rate limiting on auth endpoints (`flask-limiter`)
- [ ] Add CSRF protection for state-changing endpoints

### Environment Variables (Required for Production)

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | Flask secret key (min 32 random chars) |
| `JWT_SECRET_KEY` | JWT signing key (min 32 random chars) |
| `FLASK_ENV` | Set to `production` |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID (optional) |
| `DATABASE_PATH` | Custom SQLite path (optional) |

## Deployment

```bash
# Render build command
pip install -r requirements.txt
cd frontend && npm install && npm run build

# Render start command
gunicorn app:app --bind 0.0.0.0:$PORT
```

Set environment variables:
- `SECRET_KEY` — Flask secret key (use a strong random value)
- `JWT_SECRET_KEY` — JWT signing key (use a strong random value)
- `FLASK_ENV=production`

> **Note:** TensorFlow and PyTorch are optional for deployment. The sklearn models work without them. Add `tensorflow` and `torch` to `requirements.txt` if you want DL models in production.
