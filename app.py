import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    layout="wide"
)

st.title("Predikcija potražnje i optimizacija zaliha")
st.write("Rossmann Store Sales - Data Mining projekat")

# =====================
# Učitavanje podataka
# =====================

@st.cache_data
def load_data():
    sales = pd.read_csv(
        "data/train.csv",
        dtype={"StateHoliday": str}
    )

    stores = pd.read_csv("data/store.csv")

    df = pd.merge(
        sales,
        stores,
        on="Store",
        how="left"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )

    return df


df = load_data()

# =====================
# Sidebar
# =====================

st.sidebar.header("Podešavanja")

store_id = st.sidebar.selectbox(
    "Izaberi prodavnicu",
    sorted(df["Store"].unique())
)

lead_time = st.sidebar.slider(
    "Lead time - broj dana isporuke",
    min_value=1,
    max_value=30,
    value=7
)

ordering_cost = st.sidebar.number_input(
    "Trošak jedne narudžbe",
    min_value=1.0,
    value=50.0
)

holding_cost = st.sidebar.number_input(
    "Trošak skladištenja po jedinici",
    min_value=0.1,
    value=2.0
)

z_score = st.sidebar.selectbox(
    "Servisni nivo",
    options=[1.28, 1.65, 1.96],
    index=1,
    format_func=lambda x: {
        1.28: "90%",
        1.65: "95%",
        1.96: "97.5%"
    }[x]
)

store_data = df[
    (df["Store"] == store_id) &
    (df["Open"] == 1) &
    (df["Sales"] > 0)
].copy()

store_data = store_data.sort_values("Date")

# =====================
# Tabs
# =====================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Dashboard",
        "Forecasting",
        "Inventory Optimization",
        "Anomaly Detection",
        "What-if Simulator"
    ]
)

# =====================
# Dashboard
# =====================

with tab1:
    st.header("Pregled prodaje")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prosječna prodaja",
        f"{store_data['Sales'].mean():.2f}"
    )

    col2.metric(
        "Maksimalna prodaja",
        f"{store_data['Sales'].max():.0f}"
    )

    col3.metric(
        "Broj zapisa",
        len(store_data)
    )

    st.subheader("Prodaja kroz vrijeme")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(store_data["Date"], store_data["Sales"])
    ax.set_xlabel("Datum")
    ax.set_ylabel("Prodaja")
    ax.set_title(f"Prodaja kroz vrijeme - Store {store_id}")
    st.pyplot(fig)

    st.subheader("Prvih 10 redova")
    st.dataframe(store_data.head(10))

# =====================
# Forecasting
# =====================

with tab2:
    st.header("XGBoost predikcija prodaje")

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

    model_data = df[
        (df["Open"] == 1) &
        (df["Sales"] > 0)
    ].copy()

    X = model_data[features]
    y = model_data["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    with st.spinner("Treniranje XGBoost modela..."):
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    st.metric("MAE - prosječna apsolutna greška", f"{mae:.2f}")

    results = pd.DataFrame({
        "Stvarna prodaja": y_test.values[:20],
        "Predikcija": np.round(predictions[:20])
    })

    st.subheader("Primjer predikcija")
    st.dataframe(results)

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Važnost feature-a")
    st.bar_chart(
        importance.set_index("Feature")
    )

# =====================
# Inventory Optimization
# =====================

with tab3:
    st.header("Optimizacija zaliha")

    average_daily_demand = store_data["Sales"].mean()
    std_daily_demand = store_data["Sales"].std()
    annual_demand = average_daily_demand * 365

    safety_stock = z_score * std_daily_demand * np.sqrt(lead_time)

    demand_during_lead_time = average_daily_demand * lead_time

    reorder_point = demand_during_lead_time + safety_stock

    eoq = np.sqrt(
        (2 * annual_demand * ordering_cost) / holding_cost
    )

    recommended_order_quantity = eoq + safety_stock

    col1, col2 = st.columns(2)

    col1.metric(
        "Safety Stock",
        f"{safety_stock:.2f}"
    )

    col2.metric(
        "Reorder Point",
        f"{reorder_point:.2f}"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "EOQ",
        f"{eoq:.2f}"
    )

    col4.metric(
        "Preporučena količina narudžbe",
        f"{recommended_order_quantity:.2f}"
    )

    st.write("Parametri se mogu mijenjati u sidebar-u.")

# =====================
# Anomaly Detection
# =====================

with tab4:
    st.header("Detekcija anomalija")

    mean_sales = store_data["Sales"].mean()
    std_sales = store_data["Sales"].std()

    store_data["Z_Score"] = (
        store_data["Sales"] - mean_sales
    ) / std_sales

    store_data["Anomaly"] = store_data["Z_Score"].abs() > 2

    anomalies = store_data[store_data["Anomaly"]]

    st.metric("Broj detektovanih anomalija", len(anomalies))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        store_data["Date"],
        store_data["Sales"],
        label="Prodaja"
    )

    ax.scatter(
        anomalies["Date"],
        anomalies["Sales"],
        label="Anomalije"
    )

    ax.set_xlabel("Datum")
    ax.set_ylabel("Prodaja")
    ax.set_title(f"Anomaly Detection - Store {store_id}")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Tabela anomalija")
    st.dataframe(
        anomalies[
            [
                "Date",
                "Sales",
                "Promo",
                "StateHoliday",
                "SchoolHoliday",
                "Z_Score"
            ]
        ]
    )

    # =====================
# What-if Simulator
# =====================

with tab5:
    st.header("What-if Simulator")

    st.write(
        "Simulacija pokazuje kako bi se predikcija prodaje promijenila "
        "ako uključimo ili isključimo promociju."
    )

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

    model_data = df[
        (df["Open"] == 1) &
        (df["Sales"] > 0)
    ].copy()

    X = model_data[features]
    y = model_data["Sales"]

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    with st.spinner("Treniranje modela za What-if simulaciju..."):
        model.fit(X, y)

    latest_row = store_data.sort_values("Date").iloc[-1:].copy()

    st.subheader("Ulazni scenario")

    col1, col2, col3 = st.columns(3)

    selected_day = col1.selectbox(
        "Dan u sedmici",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=4
    )

    selected_promo = col2.selectbox(
        "Promocija",
        options=[0, 1],
        format_func=lambda x: "Bez promocije" if x == 0 else "Sa promocijom"
    )

    selected_school_holiday = col3.selectbox(
        "Školski raspust",
        options=[0, 1],
        format_func=lambda x: "Ne" if x == 0 else "Da"
    )

    today = pd.Timestamp.today()

    scenario = pd.DataFrame([{
        "Store": store_id,
        "DayOfWeek": selected_day,
        "Promo": selected_promo,
        "SchoolHoliday": selected_school_holiday,
        "Year": today.year,
        "Month": today.month,
        "Day": today.day,
        "WeekOfYear": int(today.isocalendar().week),
        "IsWeekend": 1 if selected_day in [6, 7] else 0,
        "CompetitionDistance": latest_row["CompetitionDistance"].values[0],
        "Promo2": latest_row["Promo2"].values[0]
    }])

    predicted_sales = model.predict(scenario)[0]

    st.metric(
        "Predviđena prodaja za izabrani scenario",
        f"{predicted_sales:.0f}"
    )

    st.subheader("Poređenje: bez promocije vs. sa promocijom")

    scenario_no_promo = scenario.copy()
    scenario_no_promo["Promo"] = 0

    scenario_with_promo = scenario.copy()
    scenario_with_promo["Promo"] = 1

    pred_no_promo = model.predict(scenario_no_promo)[0]
    pred_with_promo = model.predict(scenario_with_promo)[0]

    difference = pred_with_promo - pred_no_promo

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Bez promocije",
        f"{pred_no_promo:.0f}"
    )

    col2.metric(
        "Sa promocijom",
        f"{pred_with_promo:.0f}"
    )

    col3.metric(
        "Efekat promocije",
        f"{difference:.0f}"
    )

    comparison = pd.DataFrame({
        "Scenario": ["Bez promocije", "Sa promocijom"],
        "Predviđena prodaja": [
            round(pred_no_promo),
            round(pred_with_promo)
        ]
    })

    st.bar_chart(
        comparison.set_index("Scenario")
    )