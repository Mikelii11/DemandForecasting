import pandas as pd

# Ucitavanje podataka
sales = pd.read_csv(
    "../data/train.csv",
    dtype={"StateHoliday": str}
)

stores = pd.read_csv("../data/store.csv")

# Spajanje
df = pd.merge(
    sales,
    stores,
    on="Store",
    how="left"
)

# Datum u datetime
df["Date"] = pd.to_datetime(df["Date"])

# Novi feature-i
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week

# Vikend
df["IsWeekend"] = (
    df["DayOfWeek"].isin([6, 7])
).astype(int)

print(df[
    [
        "Date",
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
        "IsWeekend"
    ]
].head())

print("\nShape:")
print(df.shape)