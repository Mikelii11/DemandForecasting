import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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
# Obrada datuma
# =====================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

df["IsWeekend"] = (
    df["DayOfWeek"].isin([6, 7])
).astype(int)

# =====================
# Filtriranje
# =====================

df = df[df["Open"] == 1]
df = df[df["Sales"] > 0]

# =====================
# Nedostajuce vrijednosti
# =====================

df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
    df["CompetitionDistance"].median()
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
# Train/Test split
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================
# XGBoost model
# =====================

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================
# Predikcija
# =====================

predictions = model.predict(X_test)

# =====================
# Evaluacija
# =====================

mae = mean_absolute_error(y_test, predictions)

print("\n========================")
print("XGBOOST REZULTATI")
print("========================")

print(f"\nMAE: {mae:.2f}")

print("\nPrvih 10 predikcija:\n")

for real, pred in zip(y_test.head(10), predictions[:10]):
    print(
        f"Stvarna prodaja: {real} | Predikcija: {round(pred)}"
    )

# =====================
# Feature importance
# =====================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nVaznost feature-a:")
print(importance)