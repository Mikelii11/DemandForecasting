import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ==================================
# Učitavanje podataka
# ==================================

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

# ==================================
# Obrada datuma
# ==================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df["IsWeekend"] = (
    df["DayOfWeek"].isin([6, 7])
).astype(int)

# ==================================
# Samo otvorene prodavnice
# ==================================

df = df[df["Open"] == 1]

# ==================================
# Feature-i
# ==================================

features = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "Year",
    "Month",
    "Day",
    "IsWeekend"
]

X = df[features]

y = df["Sales"]

# ==================================
# Podjela na trening i test skup
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================
# Random Forest model
# ==================================

model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==================================
# Predikcija
# ==================================

predictions = model.predict(X_test)

# ==================================
# Evaluacija
# ==================================

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\n========================")
print("REZULTATI MODELA")
print("========================")

print(f"\nMAE: {mae:.2f}")

print("\nPrvih 10 predikcija:\n")

for real, pred in zip(
    y_test.head(10),
    predictions[:10]
):
    print(
        f"Stvarna prodaja: {real} | Predikcija: {round(pred)}"
    )