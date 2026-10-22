# 🧠 Mental Health Score Predictor (Mansik Santulan Score)

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.160+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-3.1+-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Deployment**: **[https://montana-hoped-yeast-ricky.trycloudflare.com](https://montana-hoped-yeast-ricky.trycloudflare.com)**  
> 📖 **Interactive Swagger API Docs**: **[https://montana-hoped-yeast-ricky.trycloudflare.com/docs](https://montana-hoped-yeast-ricky.trycloudflare.com/docs)**

An end-to-end Machine Learning web application and REST API designed to evaluate, understand, and predict student mental wellness scores based on digital habits, sleep cycles, academic study patterns, and lifestyle stress indicators.

Powered by a **Random Forest Regressor Pipeline** achieving an **$R^2$ score of 0.95** with a **Mean Absolute Error (MAE) under 0.19 points** across a comprehensive dataset of 5,000 students.

---

## ✨ Features

- **🎯 Machine Learning Pipeline**: Integrated `ColumnTransformer` with `log1p` transformation on skewed features (`Study_Hours`), `StandardScaler` on numerical features, `OrdinalEncoder` for stress tiers, and `OneHotEncoder` for demographic & categorical habits.
- **⚡ High-Performance REST API**: Built with FastAPI, asynchronous endpoints, and rigorous Pydantic schema validation.
- **🎨 Interactive Web Dashboard**: Editorial glassmorphic interface with an animated SVG arc gauge, live validation, and signal status indicators.
- **🎲 1-Click Dataset Integration**: Directly inspect real student records from the 5,000-entry dataset and compare the model's prediction against ground-truth values in real time.
- **🌐 Dual-Deployment Support**: Automatically defaults to local FastAPI serving with smart fallback to cloud endpoints.

---

## 📊 Model Performance & Benchmarks

The model was evaluated against the complete [Student Social Media And Mental Health Impact](Student%20Social%20Media%20And%20Mental%20Health%20Impact.csv) dataset:

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **$R^2$ Score** | **0.9500** | Explains 95% of the variance in mental health scores |
| **Mean Absolute Error (MAE)** | **0.1897** | Predictions deviate by less than 0.19 points on a 0–10 scale |
| **Root Mean Squared Error (RMSE)** | **0.2860** | Highly consistent predictions with minimal outlier variance |
| **Training/Test Split** | **70% / 30%** | Cross-validated with hyperparameter tuning |

### Prediction Samples vs. Ground Truth

| Profile Summary | Actual Score | Predicted Score | Difference |
| :--- | :---: | :---: | :---: |
| **Balanced**: 21 M, Undergraduate, 4.0h Screen, 6.7h Sleep, Stress: Medium | `6.80` | `6.75` | `-0.05` |
| **Resilient**: 23 F, Undergraduate, 5.5h Screen, 7.5h Sleep, Stress: Low | `7.60` | `7.69` | `+0.09` |
| **Strained**: 20 M, High School, 8.2h Screen, 4.8h Sleep, Stress: Very High | `4.40` | `4.49` | `+0.09` |
| **Academic**: 22 F, Graduate, 2.1h Screen, 8.2h Sleep, Stress: Low | `8.60` | `8.12` | `-0.48` |

---

## 🗂️ Dataset Architecture

The dataset (`Student Social Media And Mental Health Impact.csv`) contains **5,000 verified student records** across 13 core dimensions:

| Field | Type | Description |
| :--- | :---: | :--- |
| `Age` | Integer | Student age (10–100) |
| `Gender` | String | `Male`, `Female` |
| `Country` | String | Country of origin (top 10 preserved; others grouped) |
| `Academic_Level` | String | `High School`, `Undergraduate`, `Graduate` |
| `Most_Used_Platform`| String | `Instagram`, `TikTok`, `YouTube`, `Snapchat`, `Facebook`, etc. |
| `Purpose_Of_Use` | String | `Entertainment`, `Education`, `Networking`, `News` |
| `Avg_Daily_Usage_Hours` | Float | Screen time dedicated to social media (0–24 hrs) |
| `Daily_Unlocks` | Integer | Number of smartphone unlocks per day |
| `Study_Hours` | Float | Daily dedicated study hours |
| `Physical_Activity_Hours`| Float | Daily exercise or active hours |
| `Sleep_Hours_Per_Night` | Float | Nightly sleep duration |
| `Stress_Level` | String | `Low`, `Medium`, `High`, `Very High` |
| `Mental_Health_Score` | Float | Ground truth target wellness rating (0–10) |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vanshty220106/Mental-Health-Score.git
cd Mental-Health-Score
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the Application
Run the backend server:
```bash
python main.py
```
*Or using uvicorn directly:*
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Open in Browser
- **Web App**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health & Stats**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 🔌 API Endpoints

### `POST /predict`
Generates a predicted mental health score from input habit parameters.

**Request Body:**
```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 80,
  "study_hours": 3.0,
  "physical_activity_hours": 1.5,
  "sleep_hours_per_night": 7.0,
  "stress_level": "Medium"
}
```

**Response (200 OK):**
```json
{
  "predicted_mental_health_score": 6.18,
  "band": "Signal: balanced",
  "context": "Your rhythm looks fairly steady, with some room to recover and reset."
}
```

---

### `GET /api/sample-student`
Returns a random authentic student record from the dataset for testing and verification.

**Response (200 OK):**
```json
{
  "index": 458,
  "age": 21,
  "gender": "Male",
  "country": "USA",
  "academic_level": "Undergraduate",
  "most_used_platform": "TikTok",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 5.8,
  "daily_unlocks": 176,
  "study_hours": 1.4,
  "physical_activity_hours": 1.4,
  "sleep_hours_per_night": 5.5,
  "stress_level": "High",
  "actual_mental_health_score": 5.4
}
```

---

### `GET /api/health`
Returns the operational health, model status, and benchmark statistics.

---

## 📁 Project Structure

```
Mental-Health-Score/
├── main.py                               # FastAPI backend & model inference server
├── index.html                            # Frontend web interface
├── style.css                             # Custom modern responsive styling
├── script.js                             # Client logic, SVG gauge animator, API router
├── Mental_Health_Model.pkl               # Trained Scikit-Learn Pipeline model
├── Student Social Media And Mental...csv # 5,000-record dataset
├── ML_Project.ipynb                      # Complete EDA & training notebook
├── ML Project.html                       # HTML export of training analysis
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Git exclusion rules
└── README.md                             # Comprehensive project documentation
```

---

## 🛡️ Disclaimer

This application is built strictly for educational, informational, and analytical demonstration purposes. It is **not** a clinical diagnostic tool. If you or someone you know is experiencing mental health difficulties, please reach out to a licensed professional or trusted healthcare provider.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
