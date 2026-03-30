import streamlit as st
import pandas as pd
import time

# ================= PATH =================
DATA_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\live_results.csv"

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Risk Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Real-Time Social Engineering Risk Dashboard")
st.caption("Automatic monitoring of incoming SMS / Email / Chat messages")

# ================= AUTO REFRESH =================
refresh_rate = 5  # seconds

# ================= LOAD DATA FUNCTION =================
@st.cache_data(ttl=refresh_rate)
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return pd.DataFrame()

# ================= MAIN LOOP =================
while True:
    df = load_data()

    if df.empty:
        st.warning("Waiting for incoming messages...")
        time.sleep(refresh_rate)
        st.experimental_rerun()

    # ================= METRICS =================
    total_msgs = len(df)
    high_risk = len(df[df["risk"] == "HIGH"])
    medium_risk = len(df[df["risk"] == "MEDIUM"])
    low_risk = len(df[df["risk"] == "LOW"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📨 Total Messages", total_msgs)
    col2.metric("🔴 High Risk", high_risk)
    col3.metric("🟠 Medium Risk", medium_risk)
    col4.metric("🟢 Low Risk", low_risk)

    st.divider()

    # ================= RISK DISTRIBUTION =================
    st.subheader("📈 Risk Distribution")
    st.bar_chart(df["risk"].value_counts())

    # ================= SOURCE-WISE ANALYSIS =================
    st.subheader("📡 Source-wise Threat Analysis")
    st.bar_chart(df.groupby(["source", "risk"]).size().unstack(fill_value=0))

    # ================= LIVE LOG TABLE =================
    st.subheader("🧾 Live Detection Log")
    st.dataframe(
        df.sort_values("timestamp", ascending=False),
        use_container_width=True
    )

    time.sleep(refresh_rate)
    st.rerun()
