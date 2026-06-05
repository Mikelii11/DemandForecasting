import pandas as pd
import numpy as np

# =====================
# Ucitavanje podataka
# =====================

sales = pd.read_csv(
    "../data/train.csv",
    dtype={"StateHoliday": str}
)

# Datum
sales["Date"] = pd.to_datetime(sales["Date"])

# Koristimo samo otvorene prodavnice i prodaju vecu od 0
sales = sales[
    (sales["Open"] == 1) &
    (sales["Sales"] > 0)
]

# =====================
# Izbor prodavnice
# =====================

store_id = 1

store_data = sales[sales["Store"] == store_id]

# =====================
# Osnovni parametri
# =====================

average_daily_demand = store_data["Sales"].mean()
std_daily_demand = store_data["Sales"].std()

annual_demand = average_daily_demand * 365

# Pretpostavljeni poslovni parametri
ordering_cost = 50       # trosak jedne narudzbe
holding_cost = 2         # trosak drzanja jedne jedinice robe godisnje
lead_time = 7            # broj dana cekanja isporuke
z_score = 1.65           # servisni nivo oko 95%

# =====================
# Safety Stock
# =====================

safety_stock = z_score * std_daily_demand * np.sqrt(lead_time)

# =====================
# Reorder Point
# =====================

demand_during_lead_time = average_daily_demand * lead_time

reorder_point = demand_during_lead_time + safety_stock

# =====================
# EOQ
# =====================

eoq = np.sqrt(
    (2 * annual_demand * ordering_cost) / holding_cost
)

# =====================
# Predlozena kolicina narudzbe
# =====================

recommended_order_quantity = eoq + safety_stock

# =====================
# Ispis rezultata
# =====================

print("\n==============================")
print("INVENTORY OPTIMIZATION")
print("==============================")

print(f"\nProdavnica: {store_id}")

print(f"Prosjecna dnevna potraznja: {average_daily_demand:.2f}")
print(f"Standardna devijacija dnevne potraznje: {std_daily_demand:.2f}")
print(f"Godisnja potraznja: {annual_demand:.2f}")

print("\nParametri:")
print(f"Trosak jedne narudzbe: {ordering_cost}")
print(f"Trosak skladistenja po jedinici: {holding_cost}")
print(f"Lead time: {lead_time} dana")
print(f"Z-score: {z_score}")

print("\nRezultati:")
print(f"Safety Stock: {safety_stock:.2f}")
print(f"Reorder Point: {reorder_point:.2f}")
print(f"EOQ: {eoq:.2f}")
print(f"Preporucena kolicina za narudzbu: {recommended_order_quantity:.2f}")