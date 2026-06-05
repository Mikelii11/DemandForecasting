import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Ucitavanje podataka
# =====================

sales = pd.read_csv(
    "../data/train.csv",
    dtype={"StateHoliday": str}
)

sales["Date"] = pd.to_datetime(sales["Date"])

# Koristimo samo otvorene prodavnice
sales = sales[
    (sales["Open"] == 1) &
    (sales["Sales"] > 0)
]

# =====================
# Izbor prodavnice
# =====================

store_id = 1
store_data = sales[sales["Store"] == store_id].copy()

# Sortiranje po datumu
store_data = store_data.sort_values("Date")

# =====================
# Z-score anomaly detection
# =====================

mean_sales = store_data["Sales"].mean()
std_sales = store_data["Sales"].std()

store_data["Z_Score"] = (
    store_data["Sales"] - mean_sales
) / std_sales

# Anomalija ako je z-score veci od 2 ili manji od -2
store_data["Anomaly"] = store_data["Z_Score"].abs() > 2

anomalies = store_data[store_data["Anomaly"] == True]

# =====================
# Ispis rezultata
# =====================

print("\n==============================")
print("ANOMALY DETECTION")
print("==============================")

print(f"\nProdavnica: {store_id}")
print(f"Prosjecna prodaja: {mean_sales:.2f}")
print(f"Standardna devijacija: {std_sales:.2f}")
print(f"Broj anomalija: {len(anomalies)}")

print("\nPrvih 10 anomalija:")
print(
    anomalies[
        ["Date", "Sales", "Promo", "StateHoliday", "SchoolHoliday", "Z_Score"]
    ].head(10)
)

# =====================
# Graf
# =====================

plt.figure(figsize=(12, 5))

plt.plot(
    store_data["Date"],
    store_data["Sales"],
    label="Prodaja"
)

plt.scatter(
    anomalies["Date"],
    anomalies["Sales"],
    label="Anomalije",
    marker="o"
)

plt.title(f"Anomaly Detection - Store {store_id}")
plt.xlabel("Datum")
plt.ylabel("Prodaja")
plt.legend()
plt.tight_layout()
plt.show()