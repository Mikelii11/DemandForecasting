import pandas as pd
import matplotlib.pyplot as plt

# 1. Ucitavanje podataka
sales = pd.read_csv("../data/train.csv", dtype={"StateHoliday": str})
stores = pd.read_csv("../data/store.csv")

# 2. Spajanje tabela
df = pd.merge(sales, stores, on="Store", how="left")

# 3. Pretvaranje datuma u pravi format
df["Date"] = pd.to_datetime(df["Date"])

# 4. Osnovne informacije
print("Broj redova i kolona:")
print(df.shape)

print("\nKolone:")
print(df.columns)

print("\nPrvih 5 redova:")
print(df.head())

print("\nInformacije o datasetu:")
print(df.info())

print("\nNedostajuce vrijednosti:")
print(df.isnull().sum())

print("\nOsnovna statistika:")
print(df.describe())

# 5. Ukupna dnevna prodaja
daily_sales = df.groupby("Date")["Sales"].sum()

# 6. Graf prodaje kroz vrijeme
plt.figure(figsize=(12, 5))
daily_sales.plot()

plt.title("Ukupna dnevna prodaja")
plt.xlabel("Datum")
plt.ylabel("Prodaja")

plt.tight_layout()
plt.show()