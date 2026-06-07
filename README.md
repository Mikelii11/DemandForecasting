# Demand Forecasting and Inventory Optimization

## Project Overview

This project was developed as part of the Data Mining course.

The objective of the project is to predict future sales demand and optimize inventory management using machine learning techniques. The solution combines demand forecasting, inventory optimization, anomaly detection, explainable AI, and an interactive web dashboard.

## Dataset

Rossmann Store Sales Dataset

The dataset contains:

* Store information
* Daily sales records
* Promotions
* Holidays
* Competition information
* Customer activity

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* SHAP
* Streamlit

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed exploratory data analysis to understand the structure and quality of the dataset:

* Dataset exploration
* Missing value analysis
* Sales trend visualization
* Statistical analysis of sales data

### 2. Feature Engineering

Additional features were created from the original dataset:

* Year
* Month
* Day
* Week of Year
* Weekend indicator

These features improved the forecasting performance of machine learning models.

### 3. Demand Forecasting

Two machine learning models were implemented and compared:

#### Random Forest Regressor (Baseline Model)

Result:

* MAE = 998.18

#### XGBoost Regressor (Final Model)

Result:

* MAE = 977.34

The XGBoost model achieved lower prediction error and was selected as the final forecasting model.

### 4. Inventory Optimization

Inventory management techniques were implemented using demand statistics and forecasting results.

Implemented methods:

* Economic Order Quantity (EOQ)
* Safety Stock
* Reorder Point

Example results for Store 1:

* Average Daily Demand: 4759.10
* Safety Stock: 4418.34
* Reorder Point: 37732.01
* EOQ: 9319.52
* Recommended Order Quantity: 13737.86

These calculations help determine when and how much inventory should be ordered.

### 5. Anomaly Detection

Anomaly detection was implemented using the Z-Score method.

The system automatically identifies unusually high or unusually low sales values that may indicate:

* Promotional campaigns
* Seasonal effects
* Exceptional demand events
* Operational issues

Detected anomalies are visualized directly within the dashboard.

### 6. Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was used to explain model predictions.

The analysis identified the most influential features:

1. Promo
2. CompetitionDistance
3. Store
4. Promo2
5. DayOfWeek

The SHAP analysis improves transparency and helps understand the factors driving sales predictions.

### 7. Interactive Dashboard

A complete Streamlit dashboard was developed.

Dashboard modules:

* Dashboard
* Forecasting
* Inventory Optimization
* Anomaly Detection
* What-if Simulator

### What-if Simulation

The dashboard allows users to simulate different business scenarios and evaluate their impact on predicted sales.

Examples:

* Promotion ON/OFF
* School Holiday ON/OFF
* Different day-of-week scenarios

## Running the Project

Install required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
DemandForecasting/
│
├── data/
│
├── src/
│   ├── explore_data.py
│   ├── feature_engineering.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── inventory_optimization.py
│   ├── anomaly_detection.py
│   └── shap_analysis.py
│
├── app.py
├── requirements.txt
└── README.md
```

## Author

Miloš Vešović

Data Mining Project

2026
