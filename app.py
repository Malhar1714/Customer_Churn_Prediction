
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"

# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F5F7FB;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101D36 0%, #172C4D 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px;
}

/* Headings */
h1, h2, h3 {
    color: #172B4D;
    font-weight: 700;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(20, 40, 70, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #66758C;
    font-size: 14px;
}

[data-testid="stMetricValue"] {
    color: #172B4D;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 8px;
    padding: 12px 20px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
}

/* Custom cards */
.custom-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E5EAF2;
    margin-bottom: 15px;
}

.small-text {
    color: #748198;
    font-size: 13px;
}

.big-number {
    font-size: 28px;
    font-weight: 700;
    color: #172B4D;
}

div.stButton > button {
    background: #2457D6;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #1743AF;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 📊 ChurnIQ")

    st.markdown(
        "<p style='color:#B7C8E3;'>Customer Intelligence Platform</p>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select Dashboard",
        [
            "Overview",
            "Customer Insights",
            "Churn Prediction",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### DATA INFORMATION")

    st.write(f"Total Records: {len(df):,}")
    st.write(f"Total Features: {len(df.columns)}")

    st.divider()

    st.caption("Customer Churn Analysis Project")
    st.caption("Built with Python & Streamlit")


# --------------------------------------------------
# 5. COMMON CALCULATIONS
# --------------------------------------------------

total_customers = len(df)

churned_customers = (
    df["Churn"].eq("Yes").sum()
)

retained_customers = (
    df["Churn"].eq("No").sum()
)

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers else 0
)

retention_rate = (
    retained_customers / total_customers * 100
    if total_customers else 0
)

monthly_revenue = df["MonthlyCharges"].sum()

avg_monthly_charge = df["MonthlyCharges"].mean()

COLORS = {
    "navy": "#172B4D",
    "blue": "#3975E8",
    "red": "#E45C66",
    "green": "#19A982",
    "purple": "#8966D8",
    "gray": "#8492A6"
}

CHART_LAYOUT = {
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "font": {"color": "#34445E"},
    "margin": dict(l=20, r=20, t=55, b=20),
    "legend": {"orientation": "h", "y": -0.2}
}


# --------------------------------------------------
# 6. HEADER
# --------------------------------------------------

st.markdown(
    "<h1>Customer Churn Analytics</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:#748198;'>"
    "Monitor customer behavior, discover churn patterns "
    "and generate data-driven insights."
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.subheader("Business Overview")

    st.caption(
        "A quick snapshot of customer activity and retention."
    )

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{total_customers:,}",
        help="Total number of customer records."
    )

    c2.metric(
        "Churned Customers",
        f"{churned_customers:,}",
        f"{churn_rate:.1f}% of customers"
    )

    c3.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%"
    )

    c4.metric(
        "Monthly Charges",
        f"${monthly_revenue:,.0f}",
        help="Sum of customer monthly charges."
    )

    st.write("")

    # CHART ROW 1

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Churn Distribution")

        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Customers"]

        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            hole=0.65,
            color="Churn",
            color_discrete_map={
                "Yes": COLORS["red"],
                "No": COLORS["blue"]
            }
        )

        fig.update_layout(**CHART_LAYOUT)
        fig.update_traces(
            textinfo="percent+label",
            textfont_size=13
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Churn by Contract")

        contract_data = (
            df.groupby("Contract", as_index=False)
            .agg(
                Total=("customerID", "count"),
                Churned=(
                    "Churn",
                    lambda x: (x == "Yes").sum()
                )
            )
        )

        contract_data["Churn Rate"] = (
            contract_data["Churned"]
            / contract_data["Total"] * 100
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn Rate",
            color="Contract",
            text_auto=".1f",
            color_discrete_sequence=[
                "#3975E8", "#19A982", "#8966D8"
            ],
            title="Percentage of Customers Churning"
        )

        fig.update_layout(**CHART_LAYOUT)
        fig.update_yaxes(title="Churn Rate (%)")
        fig.update_xaxes(title="Contract Type")

        st.plotly_chart(fig, use_container_width=True)

    st.write("")

    # CHART ROW 2

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Customer Tenure Distribution")

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={
                "Yes": COLORS["red"],
                "No": COLORS["blue"]
            }
        )

        fig.update_layout(**CHART_LAYOUT)
        fig.update_xaxes(title="Tenure (Months)")
        fig.update_yaxes(title="Number of Customers")

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        st.subheader("Monthly Charges by Churn")

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={
                "Yes": COLORS["red"],
                "No": COLORS["blue"]
            }
        )

        fig.update_layout(**CHART_LAYOUT)
        fig.update_xaxes(title="Churn Status")
        fig.update_yaxes(title="Monthly Charges ($)")

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Quick Business Insights")

    short_contract = contract_data.sort_values(
        "Churn Rate", ascending=False
    ).iloc[0]

    st.info(
        f"**Highest churn contract:** {short_contract['Contract']} "
        f"with a churn rate of {short_contract['Churn Rate']:.2f}%."
    )

    st.info(
        f"**Average monthly charges:** "
        f"${avg_monthly_charge:.2f} per customer."
    )


# ==================================================
# PAGE 2: CUSTOMER INSIGHTS
# ==================================================

elif page == "Customer Insights":

    st.subheader("Customer Segmentation & Analysis")

    st.caption(
        "Explore customer behavior using interactive filters."
    )

    # FILTERS

    f1, f2, f3 = st.columns(3)

    with f1:

        contract_filter = st.selectbox(
            "Contract Type",
            ["All"] + sorted(df["Contract"].dropna().unique())
        )

    with f2:

        gender_filter = st.selectbox(
            "Gender",
            ["All"] + sorted(df["gender"].dropna().unique())
        )

    with f3:

        payment_filter = st.selectbox(
            "Payment Method",
            ["All"] + sorted(
                df["PaymentMethod"].dropna().unique()
            )
        )

    filtered = df.copy()

    if contract_filter != "All":
        filtered = filtered[
            filtered["Contract"] == contract_filter
        ]

    if gender_filter != "All":
        filtered = filtered[
            filtered["gender"] == gender_filter
        ]

    if payment_filter != "All":
        filtered = filtered[
            filtered["PaymentMethod"] == payment_filter
        ]

    st.write("")

    a, b, c = st.columns(3)

    a.metric("Filtered Customers", f"{len(filtered):,}")

    filtered_churn = (
        filtered["Churn"].eq("Yes").mean() * 100
        if len(filtered) else 0
    )

    b.metric("Filtered Churn Rate", f"{filtered_churn:.2f}%")

    c.metric(
        "Average Tenure",
        f"{filtered['tenure'].mean():.1f} months"
        if len(filtered) else "N/A"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Churn by Internet Service")

        internet_data = (
            filtered.groupby("InternetService", as_index=False)
            .agg(
                Customers=("customerID", "count"),
                Churned=(
                    "Churn",
                    lambda x: (x == "Yes").sum()
                )
            )
        )

        internet_data["Churn Rate"] = (
            internet_data["Churned"]
            / internet_data["Customers"] * 100
        )

        fig = px.bar(
            internet_data,
            x="InternetService",
            y="Churn Rate",
            color="InternetService",
            text_auto=".1f"
        )

        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Churn by Payment Method")

        payment_data = (
            filtered.groupby("PaymentMethod", as_index=False)
            .agg(
                Customers=("customerID", "count"),
                Churned=(
                    "Churn",
                    lambda x: (x == "Yes").sum()
                )
            )
        )

        payment_data["Churn Rate"] = (
            payment_data["Churned"]
            / payment_data["Customers"] * 100
        )

        fig = px.bar(
            payment_data,
            x="PaymentMethod",
            y="Churn Rate",
            color="PaymentMethod",
            text_auto=".1f"
        )

        fig.update_layout(**CHART_LAYOUT)
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Customer Data Explorer")

    st.dataframe(
        filtered,
        use_container_width=True,
        height=350
    )

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Filtered Customer Data",
        data=csv,
        file_name="filtered_customers.csv",
        mime="text/csv"
    )


# ==================================================
# PAGE 3: CHURN PREDICTION
# ==================================================

elif page == "Churn Prediction":

    st.subheader("Customer Churn Prediction")

    st.caption(
        "Enter customer details to estimate churn probability."
    )

    if model is None:

        st.warning(
            "Model file not found. Train your model and save it "
            "to the models/churn_model.joblib location."
        )

    else:

        with st.form("prediction_form"):

            st.markdown("### Customer Information")

            col1, col2, col3 = st.columns(3)

            with col1:

                gender = st.selectbox(
                    "Gender", ["Male", "Female"]
                )

                senior = st.selectbox(
                    "Senior Citizen", [0, 1]
                )

                partner = st.selectbox(
                    "Has Partner", ["Yes", "No"]
                )

                dependents = st.selectbox(
                    "Has Dependents", ["Yes", "No"]
                )

            with col2:

                tenure = st.slider(
                    "Tenure (Months)", 0, 72, 12
                )

                phone = st.selectbox(
                    "Phone Service", ["Yes", "No"]
                )

                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    ["Yes", "No", "No phone service"]
                )

                internet = st.selectbox(
                    "Internet Service",
                    ["DSL", "Fiber optic", "No"]
                )

            with col3:

                contract = st.selectbox(
                    "Contract",
                    ["Month-to-month", "One year", "Two year"]
                )

                paperless = st.selectbox(
                    "Paperless Billing", ["Yes", "No"]
                )

                payment = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )

                monthly = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=200.0,
                    value=70.0
                )

            st.markdown("### Service Information")

            col4, col5 = st.columns(2)

            with col4:

                security = st.selectbox(
                    "Online Security",
                    ["Yes", "No", "No internet service"]
                )

                backup = st.selectbox(
                    "Online Backup",
                    ["Yes", "No", "No internet service"]
                )

                protection = st.selectbox(
                    "Device Protection",
                    ["Yes", "No", "No internet service"]
                )

                support = st.selectbox(
                    "Tech Support",
                    ["Yes", "No", "No internet service"]
                )

            with col5:

                tv = st.selectbox(
                    "Streaming TV",
                    ["Yes", "No", "No internet service"]
                )

                movies = st.selectbox(
                    "Streaming Movies",
                    ["Yes", "No", "No internet service"]
                )

            total_charges = monthly * tenure

            st.caption(
                f"Estimated Total Charges: ${total_charges:.2f}"
            )

            submitted = st.form_submit_button(
                "🔍 Predict Customer Churn",
                use_container_width=True
            )

        if submitted:

            input_data = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total_charges
            }])

            try:

                probability = model.predict_proba(
                    input_data
                )[0][1]

                prediction = model.predict(input_data)[0]

                st.divider()

                st.subheader("Prediction Result")

                r1, r2 = st.columns(2)

                with r1:

                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(float(probability))

                with r2:

                    if prediction == 1:
                        st.error(
                            "⚠️ Model predicts: Likely to Churn"
                        )
                    else:
                        st.success(
                            "✅ Model predicts: Likely to Stay"
                        )

                st.caption(
                    "This is a model estimate, not a guarantee "
                    "of future customer behavior."
                )

            except Exception as e:

                st.error(f"Prediction error: {e}")


# ==================================================
# PAGE 4: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.subheader("Machine Learning Model Performance")

    st.caption(
        "Review the performance of your trained churn prediction model."
    )

    if model is None:

        st.warning(
            "Model file is missing. Please train and save your model."
        )

    else:

        st.success("Trained machine learning model loaded successfully.")

        st.markdown(
            """
            **Model:** Saved Scikit-learn pipeline

            **Problem type:** Binary Classification

            **Target variable:** Churn

            **Classes:** 0 = No Churn, 1 = Churn
            """
        )

        st.info(
            "For actual evaluation metrics, use the held-out test data "
            "from your model training notebook. Do not evaluate on "
            "training data and present those numbers as test performance."
        )

        st.markdown(
            "Open your model training notebook to view the model "
            "comparison table, classification report and ROC curve."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8492A6; padding:15px;">
        Customer Churn Analytics Dashboard |
        Built with Python, Streamlit & Plotly
        <br>
        Data Science Portfolio Project
    </div>
    """,
    unsafe_allow_html=True
)
