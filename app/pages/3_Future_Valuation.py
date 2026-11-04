
import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="Future Property Valuation",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 Five-Year Property Valuation")

st.markdown(
    """
    Explore possible future property values using adjustable
    annual appreciation assumptions.
    """
)

st.subheader("Valuation Methodology")

st.latex(r"FV = PV \times (1+r)^t")

st.write(
    """
    - FV: Future property value
    - PV: Current property value
    - r: Annual appreciation rate
    - t: Investment period in years

    This is a compound-growth scenario calculator, not a
    machine-learning price prediction.
    """
)

st.divider()

st.subheader("Investment Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    current_value = st.number_input(
        "Current Property Value (₹ Lakhs)",
        min_value=1.0,
        max_value=100000.0,
        value=100.0,
        step=5.0,
    )

with col2:
    annual_rate = st.slider(
        "Annual Appreciation (%)",
        min_value=0.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
    )

with col3:
    years = st.slider(
        "Investment Period (Years)",
        min_value=1,
        max_value=30,
        value=5,
    )

selected_rate = annual_rate / 100

future_value = current_value * (1 + selected_rate) ** years
total_appreciation = future_value - current_value
appreciation_percentage = (
    total_appreciation / current_value
) * 100

st.divider()

st.subheader("Your Valuation Scenario")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Property Value",
    f"₹{current_value:,.2f} Lakhs",
)

col2.metric(
    "Estimated Future Value",
    f"₹{future_value:,.2f} Lakhs",
    delta=f"₹{total_appreciation:,.2f} Lakhs",
)

col3.metric(
    "Total Appreciation",
    f"{appreciation_percentage:.2f}%",
)

st.divider()

st.subheader("Scenario Comparison")

scenario_rates = {
    "Conservative": 4.0,
    "Moderate": 8.0,
    "Optimistic": 12.0,
}

scenario_data = []

for scenario, rate in scenario_rates.items():
    projected_value = (
        current_value * (1 + rate / 100) ** years
    )

    scenario_data.append({
        "Scenario": scenario,
        "Annual Appreciation": rate,
        "Future Value": projected_value,
        "Total Appreciation": projected_value - current_value,
    })

scenario_df = pd.DataFrame(scenario_data)

display_df = scenario_df.copy()
display_df["Annual Appreciation"] = (
    display_df["Annual Appreciation"].map(lambda x: f"{x:.1f}%")
)
display_df["Future Value"] = (
    display_df["Future Value"].map(lambda x: f"₹{x:,.2f} Lakhs")
)
display_df["Total Appreciation"] = (
    display_df["Total Appreciation"].map(
        lambda x: f"₹{x:,.2f} Lakhs"
    )
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

scenario_chart = px.bar(
    scenario_df,
    x="Scenario",
    y="Future Value",
    color="Scenario",
    text="Future Value",
    title="Future Value Under Different Appreciation Scenarios",
)

scenario_chart.update_traces(
    texttemplate="₹%{text:.2f} L",
    textposition="outside",
)

st.plotly_chart(
    scenario_chart,
    use_container_width=True,
)

st.divider()

st.subheader("Year-wise Value Projection")

growth_data = []

for year in range(years + 1):
    projected_value = (
        current_value * (1 + selected_rate) ** year
    )

    growth_data.append({
        "Year": year,
        "Property Value": projected_value,
    })

growth_df = pd.DataFrame(growth_data)

growth_chart = px.line(
    growth_df,
    x="Year",
    y="Property Value",
    markers=True,
    title="Projected Property Value Over Time",
)

st.plotly_chart(
    growth_chart,
    use_container_width=True,
)

st.dataframe(
    growth_df.assign(
        **{
            "Property Value": growth_df["Property Value"].map(
                lambda x: f"₹{x:,.2f} Lakhs"
            )
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.warning(
    """
    **Important financial limitation**

    These are illustrative calculations based on assumed
    appreciation rates. They are not verified market forecasts
    or guaranteed investment returns.

    The calculation excludes taxes, registration fees,
    maintenance, loan interest, transaction costs, rental
    income, inflation and market volatility.
    """
)
