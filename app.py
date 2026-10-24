import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

st.title("💰 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")
st.write("Calculate your EMI and explore your estimated financial risk.")

# Load trained machine learning model
try:
    model = joblib.load("emi_risk_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

st.header("🏦 Loan Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Monthly Income (₹)",
        min_value=1000,
        value=50000,
        step=5000
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=1000,
        value=500000,
        step=10000
    )

    interest_rate = st.number_input(
        "Annual Interest Rate (%)",
        min_value=0.0,
        value=8.5,
        step=0.5
    )

with col2:
    tenure_years = st.number_input(
        "Loan Tenure (Years)",
        min_value=1,
        max_value=40,
        value=5
    )

    existing_emi = st.number_input(
        "Existing Monthly EMIs (₹)",
        min_value=0,
        value=0,
        step=1000
    )

# Calculate monthly EMI
months = tenure_years * 12
monthly_rate = interest_rate / 1200

if monthly_rate == 0:
    emi = loan_amount / months
else:
    emi = (
        loan_amount * monthly_rate
        * (1 + monthly_rate) ** months
        / ((1 + monthly_rate) ** months - 1)
    )

st.divider()
st.header("📊 Your Results")

c1, c2, c3 = st.columns(3)

c1.metric("Monthly EMI", f"₹{emi:,.2f}")
c2.metric("Total Repayment", f"₹{emi * months:,.2f}")
c3.metric(
    "Total Interest",
    f"₹{(emi * months) - loan_amount:,.2f}"
)
# Financial burden analysis
st.divider()
st.header("📈 Financial Burden Analysis")

total_monthly_commitment = emi + existing_emi
emi_ratio = total_monthly_commitment / income

col4, col5 = st.columns(2)

with col4:
    st.metric(
        "Total Monthly EMI Burden",
        f"₹{total_monthly_commitment:,.2f}"
    )

with col5:
    st.metric(
        "EMI-to-Income Ratio",
        f"{emi_ratio:.1%}"
    )

st.progress(min(emi_ratio, 1.0))
st.subheader("💸 Monthly Income Breakdown")

remaining_income = income - total_monthly_commitment

chart_data = pd.DataFrame({
    "Category": ["Total EMI Burden", "Remaining Income"],
    "Amount (₹)": [
        total_monthly_commitment,
        max(remaining_income, 0)
    ]
})

st.bar_chart(
    chart_data,
    x="Category",
    y="Amount (₹)"
)

if remaining_income < 0:
    st.error("Your monthly commitments exceed your income.")
else:
    st.metric("Income Left After EMIs", f"₹{remaining_income:,.2f}")
    st.divider()
st.header("📋 Loan Repayment Summary")

summary = pd.DataFrame({
    "Financial Detail": [
        "Principal Loan Amount",
        "Annual Interest Rate",
        "Loan Tenure",
        "Monthly EMI",
        "Total Repayment",
        "Total Interest Payable"
    ],
    "Value": [
        f"₹{loan_amount:,.2f}",
        f"{interest_rate:.2f}%",
        f"{tenure_years} years",
        f"₹{emi:,.2f}",
        f"₹{emi * months:,.2f}",
        f"₹{(emi * months) - loan_amount:,.2f}"
    ]
})

st.dataframe(summary, hide_index=True, use_container_width=True)
st.download_button(
    label="📥 Download Loan Report",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="EMIPredict_Loan_Report.csv",
    mime="text/csv"
)


if emi_ratio <= 0.30:
    st.info("Your monthly EMI burden is relatively low.")
elif emi_ratio <= 0.50:
    st.warning("Your monthly EMI burden is moderate.")
else:
    st.error("Your monthly EMI burden is high.")

if st.button("Predict Financial Risk", type="primary"):
    features = pd.DataFrame([{
        "income": income,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "tenure_years": tenure_years,
        "existing_emi": existing_emi,
        "monthly_emi": round(emi, 2)
    }])

    risk = model.predict(features)[0]
    emi_ratio = (emi + existing_emi) / income

    st.subheader("Estimated Risk Category")

    if risk == "Low":
        st.success(f"Estimated Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Estimated Risk: {risk}")
    else:
        st.error(f"Estimated Risk: {risk}")

    st.write(f"EMI-to-income ratio: {emi_ratio:.1%}")

    st.caption(
        "Educational demonstration using synthetic training data. "
        "This prediction is not a validated credit assessment or "
        "a guarantee of loan approval."
    )
