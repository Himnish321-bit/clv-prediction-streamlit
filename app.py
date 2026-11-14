import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CLV Prediction", page_icon="🚗", layout="wide")


# Load the saved model and the data only once (not on every click)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")


model = load_model()
df = load_data()

page = st.sidebar.radio("Go to", ["1. Business Problem", "2. Data Insights", "3. Prediction"])

# ---------------------------------------------------------------- Page 1
if page == "1. Business Problem":
    st.title("Customer Lifetime Value (CLV) Prediction")
    st.subheader("Auto Insurance | MLBA Assignment 2")

    st.markdown("### Business Problem")
    st.write(
        "An auto insurance company has a limited marketing and retention budget. "
        "It does not know which customers will be worth the most over the long run, "
        "so money may be spent on the wrong customers."
    )
    st.markdown("### Analytics Objective")
    st.write("Build a regression model that predicts the Customer Lifetime Value (in dollars) of a customer.")
    st.markdown("### Business Decision")
    st.write(
        "Customers with a high predicted CLV get retention offers and cross-selling. "
        "Customers with a low predicted CLV get low-cost automated marketing."
    )
    st.markdown("### Dataset Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Average CLV", f"${df['Customer Lifetime Value'].mean():,.0f}")
    st.write(
        "**Name:** Auto Insurance Customer Lifetime Value (IBM Watson Marketing Customer Value Analysis)  \n"
        "**Source:** Public dataset available on Kaggle  \n"
        "**Target variable:** Customer Lifetime Value (net profit from a customer over their lifetime)  \n"
        "**Model used:** Random Forest Regressor (R² about 0.69, average error about $1,500)"
    )
    st.dataframe(df.head(10))

# ---------------------------------------------------------------- Page 2
elif page == "2. Data Insights":
    st.title("Data Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average CLV", f"${df['Customer Lifetime Value'].mean():,.0f}")
    c2.metric("Median CLV", f"${df['Customer Lifetime Value'].median():,.0f}")
    c3.metric("Highest CLV", f"${df['Customer Lifetime Value'].max():,.0f}")

    st.markdown("### Average CLV by Vehicle Class")
    order = df.groupby("Vehicle Class")["Customer Lifetime Value"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="Vehicle Class", y="Customer Lifetime Value", order=order, errorbar=None, ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)
    st.info("Luxury SUV and Luxury Car owners are worth more than double the owners of Four-Door and Two-Door cars. "
            "Premium service should focus on the luxury segment.")

    st.markdown("### Average CLV by Coverage")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Coverage", y="Customer Lifetime Value",
                order=["Basic", "Extended", "Premium"], errorbar=None, ax=ax)
    st.pyplot(fig)
    st.info("CLV rises with coverage level. Most customers are on Basic, so moving them up is an upselling chance.")

    st.markdown("### Monthly Premium vs CLV")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="Monthly Premium Auto", y="Customer Lifetime Value", alpha=0.3, ax=ax)
    st.pyplot(fig)
    st.info("Customers who pay a higher monthly premium tend to have a higher CLV.")

# ---------------------------------------------------------------- Page 3
else:
    st.title("Predict Customer Lifetime Value")
    st.write("Enter the customer details and click **Predict**.")

    col1, col2 = st.columns(2)
    with col1:
        premium = st.number_input("Monthly Premium Auto ($)", min_value=50, max_value=300, value=90)
        policies = st.slider("Number of Policies", 1, 9, 2)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=150000, value=40000, step=1000)
        claim = st.number_input("Total Claim Amount ($)", min_value=0, max_value=3000, value=400)
    with col2:
        last_claim = st.slider("Months Since Last Claim", 0, 35, 12)
        inception = st.slider("Months Since Policy Inception", 0, 99, 24)
        coverage = st.selectbox("Coverage", ["Basic", "Extended", "Premium"])
        vehicle = st.selectbox("Vehicle Class", sorted(df["Vehicle Class"].unique()))

    if st.button("Predict"):
        new_customer = pd.DataFrame([{
            "Monthly Premium Auto": premium,
            "Number of Policies": policies,
            "Income": income,
            "Total Claim Amount": claim,
            "Months Since Last Claim": last_claim,
            "Months Since Policy Inception": inception,
            "Coverage": coverage,
            "Vehicle Class": vehicle,
        }])
        prediction = model.predict(new_customer)[0]
        avg = df["Customer Lifetime Value"].mean()

        # Where does this customer stand compared with all customers?
        percentile = (df["Customer Lifetime Value"] < prediction).mean() * 100

        st.success(f"Predicted Customer Lifetime Value: ${prediction:,.0f}")
        st.write(f"This is about **{prediction / avg:.1f}x** the average CLV (${avg:,.0f}) "
                 f"and higher than **{percentile:.0f}%** of customers in the data.")
        st.caption("Typical error of the model is about $1,500, so treat this as an estimate.")

        # Recommended action
        if percentile >= 80:
            st.markdown("### Recommended action: HIGH VALUE")
            st.write("Give retention offers, a dedicated agent and cross-sell extra policies or higher coverage.")
        elif percentile >= 40:
            st.markdown("### Recommended action: MEDIUM VALUE")
            st.write("Keep regular service. Offer an upgrade in coverage to grow the value.")
        else:
            st.markdown("### Recommended action: LOW VALUE")
            st.write("Use low-cost, automated marketing (email, app messages). Avoid expensive retention offers.")

        # Small chart: this customer vs all customers
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.histplot(df["Customer Lifetime Value"], bins=50, ax=ax)
        ax.axvline(prediction, color="red", linestyle="--", label="This customer")
        ax.legend()
        ax.set_xlim(0, 40000)
        st.pyplot(fig)
