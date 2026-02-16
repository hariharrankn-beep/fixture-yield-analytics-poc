"""Streamlit dashboard (minimal) for Fixture Yield Analytics.

This is a small interactive UI that uses the analytics module to display basic KPIs and charts.
"""
rom __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import plotly.express as px
import pandas as pd
import io

from backend import analytics


PAGES = [
    "Executive Overview",
    "Fixture Lifecycle",
    "Machine–Fixture Analysis",
    "Vendor Performance",
]


def _get_sidebar_page() -> str:
    # If a click-navigation set a one-time page key, use it as the initial selection
    clicked = None
    if st.session_state.get("page_by_click"):
        clicked = st.session_state.pop("page_by_click")
    # Determine default index: clicked -> session_state.page -> default 0
    default = 0
    if clicked in PAGES:
        default = PAGES.index(clicked)
    else:
        cur = st.session_state.get("page")
        if cur in PAGES:
            default = PAGES.index(cur)

    # show the persistent sidebar selector; do NOT overwrite session_state['page'] here
    selected = st.sidebar.selectbox("Page", PAGES, index=default, key="sidebar_page")
    return selected


def render_executive_overview(data: dict, meta: dict) -> None:
    # Data status and production overview (kept same analytics and charts)
    if meta.get("warnings"):
        st.header("Executive Overview")
        for w in meta.get("warnings", []):
            st.warning(w)
        if meta.get("sample_generated"):
            st.info("Sample data generated automatically.")
    else:
        st.header("Executive Overview")

    fixtures: pd.DataFrame = data.get("fixtures") if data.get("fixtures") is not None else pd.DataFrame()
    prod: pd.DataFrame = data.get("production") if data.get("production") is not None else pd.DataFrame()

    st.subheader("Production Overview")
    if prod is None or prod.empty:
        st.info("No production data available.")
    else:
        agg = analytics.aggregate_production(prod)
        if agg is None or agg.empty:
            st.info("No production aggregates available.")
        else:
            st.metric("Fixtures (count)", len(agg))
            overall_yield = agg["yield"].mean()
            st.metric("Average Yield (%)", f"{overall_yield:.2f}")
            top = agg.sort_values("yield", ascending=False).head(10)
            fig = px.bar(top, x="fixture_id", y="yield", title="Top 10 Fixtures by Yield")
            st.plotly_chart(fig, use_container_width=True)

    # Fixtures table + download
    st.subheader("Fixtures")
    if fixtures is None or fixtures.empty:
        st.info("No fixture metadata available.")
    else:
        csv_buf = io.StringIO()
        fixtures.to_csv(csv_buf, index=False)
        st.download_button(label="Download fixtures CSV", data=csv_buf.getvalue(), file_name="fixtures.csv", mime="text/csv")
        with st.expander("Show all fixtures"):
            st.dataframe(fixtures)


def render_fixture_lifecycle(data: dict, meta: dict) -> None:
    st.header("Fixture Lifecycle")
    fixtures: pd.DataFrame = data.get("fixtures") if data.get("fixtures") is not None else pd.DataFrame()
    prod: pd.DataFrame = data.get("production") if data.get("production") is not None else pd.DataFrame()
    qual: pd.DataFrame = data.get("quality") if data.get("quality") is not None else pd.DataFrame()

    if fixtures.empty:
        st.info("No fixture metadata available.")
        return

    # Show a fixture selector and timeseries
    st.markdown(f"**Total fixtures:** {len(fixtures)}")
    with st.expander("Show all fixtures"):
        st.dataframe(fixtures)

    fid = st.selectbox("Select fixture", options=fixtures["fixture_id"].tolist())
    ts = analytics.compute_fixture_timeseries(fid, prod, qual)
    st.write(f"Timeseries for {fid}")
    if ts.empty:
        st.info("No timeseries data for selected fixture.")
    else:
        # show yield and mean deviation series if available
        if "yield_pct" in ts.columns:
            st.line_chart(ts.set_index("date")["yield_pct"])
        if "mean_dev_a" in ts.columns:
            st.line_chart(ts.set_index("date")["mean_dev_a"])


def render_machine_fixture(data: dict, meta: dict) -> None:
    st.header("Machine–Fixture Analysis")
    prod: pd.DataFrame = data.get("production") if data.get("production") is not None else pd.DataFrame()
    if prod.empty:
        st.info("No production data available.")
        return
    # quick pivot of produced by machine
    pivot = prod.groupby(["machine_id"]).agg(total_produced=("produced_count", "sum"), total_rejects=("reject_count", "sum")).reset_index()
    st.dataframe(pivot)


def render_vendor_performance(data: dict, meta: dict) -> None:
    st.header("Vendor Performance")
    fixtures: pd.DataFrame = data.get("fixtures") if data.get("fixtures") is not None else pd.DataFrame()
    prod: pd.DataFrame = data.get("production") if data.get("production") is not None else pd.DataFrame()
    qual: pd.DataFrame = data.get("quality") if data.get("quality") is not None else pd.DataFrame()
    vp = analytics.compute_vendor_performance(fixtures, prod, qual)
    if vp.empty:
        st.info("No vendor performance data available.")
    else:
        st.dataframe(vp)


def main() -> None:
    st.set_page_config(page_title="Fixture Yield Analytics", layout="wide")
    st.title("Fixture Yield Analytics (Minimal)")

    # load data once
    data = analytics.load_data()
    meta = data.get("__meta__", {})

    # page selection (sidebar). This selectbox is always visible and persistent.
    page = _get_sidebar_page()

    # Render selected page
    if page == "Executive Overview":
        render_executive_overview(data, meta)
    elif page == "Fixture Lifecycle":
        render_fixture_lifecycle(data, meta)
    elif page == "Machine–Fixture Analysis":
        render_machine_fixture(data, meta)
    elif page == "Vendor Performance":
        render_vendor_performance(data, meta)


if __name__ == "__main__":
    main()
