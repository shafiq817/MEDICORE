# 🏥 MediCore – Healthcare Management System

A full-stack healthcare dashboard with Python backend and modern HTML/CSS/JS frontend.

## Dataset (Kaggle-based)
Synthetic dataset based on Kaggle healthcare datasets:
- [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**500 patient records** with: age, gender, blood type, department, medical condition, vitals (BP, glucose, cholesterol, heart rate), billing, insurance, length of stay, outcomes.

## Features
| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs + 5 charts (monthly trends, dept breakdown, outcomes, insurance, billing) |
| 👥 Patients | Paginated table with search & department filter |
| 💓 Vitals & Analytics | Population health metrics, conditions, age distribution |
| 🎯 Risk Predictor | ML-powered (Random Forest) patient risk assessment |
| 🩺 Doctors | Staff performance metrics |

## Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Chart.js
- **Backend**: Python, Flask, Pandas, NumPy, Scikit-learn (Random Forest)
- **ML**: RandomForestClassifier for risk prediction

## Setup & Run

### 1. Install dependencies
```bash
pip install flask pandas numpy scikit-learn
```

### 2. Run the app
```bash
cd hms
python app.py
```

### 3. Open browser
Visit: **http://localhost:5050**

## Project Structure
```
hms/
├── app.py          # Flask backend (APIs + ML model)
├── index.html      # Frontend (dashboard, charts, tables)
├── patients.csv    # Generated healthcare dataset (500 records)
└── README.md
```
