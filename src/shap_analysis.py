import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =====================
# Ucitavanje podataka
# =====================

sales = pd.read_csv(
    "../data/train.csv",
    dtype={"StateHoliday": str}
)

stores = pd.read_csv("../data/store.csv")

df = pd.merge(
    sales,
    stores,
    on="Store",
    how="left"
)

# =====================
# Feature engineering
# =====================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

df["IsWeekend"] = (
    df["DayOfWeek"].isin([6, 7])
).astype(int)

df = df[df["Open"] == 1]
df = df[df["Sales"] > 0]

df["CompetitionDistance"] = (
    df["CompetitionDistance"]
    .fillna(df["CompetitionDistance"].median())
)

# =====================
# Feature-i
# =====================

features = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend",
    "CompetitionDistance",
    "Promo2"
]

X = df[features]
y = df["Sales"]

# =====================
# Manji uzorak
# =====================

sample_size = 50000

X = X.sample(
    sample_size,
    random_state=42
)

y = y.loc[X.index]

# =====================
# Train/Test
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================
# XGBoost
# =====================

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================
# SHAP
# =====================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# =====================
# Summary Plot
# =====================

shap.summary_plot(
    shap_values,
    X_test,
    show=True
)