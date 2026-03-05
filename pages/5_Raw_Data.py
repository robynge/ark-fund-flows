"""Raw data explorer."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_all_etfs, add_returns, ETF_NAMES

st.set_page_config(page_title="Raw Data", layout="wide")
st.title("Raw Data Explorer")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)


@st.cache_data
def get_data():
    df = load_all_etfs()
    df = add_returns(df)
    return df


df = get_data()
etf_df = df[df["ETF"] == selected_etf].copy()

st.subheader(f"{selected_etf}: {len(etf_df):,} rows")

# Date range filter
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start Date", value=etf_df["Date"].min())
with col2:
    end = st.date_input("End Date", value=etf_df["Date"].max())

mask = (etf_df["Date"].dt.date >= start) & (etf_df["Date"].dt.date <= end)
filtered = etf_df[mask].sort_values("Date", ascending=False)

st.dataframe(
    filtered[["Date", "Fund_Flow", "Open", "High", "Low", "Close", "Volume", "Return"]].style.format({
        "Fund_Flow": "{:.2f}",
        "Open": "{:.2f}",
        "High": "{:.2f}",
        "Low": "{:.2f}",
        "Close": "{:.2f}",
        "Volume": "{:,.0f}",
        "Return": "{:.4%}",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
    height=600,
)

st.download_button(
    "Download CSV",
    filtered.to_csv(index=False),
    file_name=f"{selected_etf}_data.csv",
    mime="text/csv",
)
