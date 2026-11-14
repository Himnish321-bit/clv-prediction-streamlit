import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CLV Predictor", page_icon="🚗", layout="wide")

TEAL = "#0F766E"
ORANGE = "#F59E0B"
sns.set_theme(style="whitegrid")

# ---------- Simple styling (CSS) ----------
st.markdown("""
<style>
.hero {background: linear-gradient(90deg, #0F766E, #14B8A6); padding: 28px 32px;
       border-radius: 14px; color: white; margin-bottom: 20px;}
.hero h1 {color: white; margin: 0; font-size: 2.2rem;}
.hero p {color: #E6FFFA; margin: 6px 0 0 0; font-size: 1.05rem;}
.card {background: white; padding: 18px 20px; border-radius: 12px;
       border-left: 6px solid #0F766E; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 12px;}
.card h4 {margin: 0 0 6px 0; color: #0F766E;}
.card p {margin: 0; color: #334155;}
.result {padding: 24px; border-radius: 14px; text-align: center; color: white; margin: 10px 0;}
.result h1 {color: white; margin: 0; font-size: 3rem;}
.result p {margin: 4px 0 0 0; font-size: 1.1rem;}
.high {background: linear-gradient(90deg, #059669, #10B981);}
.medium {background: linear-gradient(90deg, #D97706, #F59E0B);}
.low {background: linear-gradient(90deg, #475569, #64748B);}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title, text):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


# Load the saved model and the data only once (not on every click)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")


model = load_model()
df = load_data()

st.sidebar.markdown("## 🚗 CLV Predictor")
st.sidebar.caption("MLBA Assignment 2")
page = st.sidebar.radio("Go to", ["🏢 Business Problem", "📊 Data Insights", "🔮 Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("Model: Random Forest\n\nR² about 0.69 | Avg error about $1,500")

# ---------------------------------------------------------------- Page 1
if page == "🏢 Business Problem":
    hero("Customer Lifetime Value Prediction",
         "Auto insurance | Find your most valuable customers early")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Average CLV", f"${df['Customer Lifetime Value'].mean():,.0f}")
    c4.metric("Model R²", "0.69")

    left, right = st.columns(2)
    with left:
        card("❓ Business Problem",
             "An auto insurance company has a limited marketing and retention budget. "
             "It does not know which customers will be worth the most over time, "
             "so money may be spent on the wrong customers.")
        card("🎯 Analytics Objective",
             "Build a regression model that predicts the Customer Lifetime Value (in dollars) of a customer.")
    with right:
        card("✅ Business Decision",
             "Customers with a high predicted CLV get retention offers and cross-selling. "
             "Customers with a low predicted CLV get low-cost automated marketing.")
        card("🗂️ Dataset",
             "Auto Insurance Customer Lifetime Value (IBM Watson Marketing Customer Value Analysis), "
             "public dataset on Kaggle. Target: Customer Lifetime Value.")

    st.markdown("#### Sample of the data")
    st.dataframe(df.head(10), width="stretch")

# ---------------------------------------------------------------- Page 2
elif page == "📊 Data Insights":
    hero("Data Insights", "What the data tells the business manager")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average CLV", f"${df['Customer Lifetime Value'].mean():,.0f}")
    c2.metric("Median CLV", f"${df['Customer Lifetime Value'].median():,.0f}")
    c3.metric("Highest CLV", f"${df['Customer Lifetime Value'].max():,.0f}")

    tab1, tab2, tab3 = st.tabs(["🚙 Vehicle Class", "🛡️ Coverage", "💵 Premium"])

    with tab1:
        order = df.groupby("Vehicle Class")["Customer Lifetime Value"].mean().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df, x="Vehicle Class", y="Customer Lifetime Value", order=order,
                    errorbar=None, color=TEAL, ax=ax)
        ax.set_ylabel("Average CLV ($)")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        st.info("Luxury SUV and Luxury Car owners are worth more than double the owners of Four-Door "
                "and Two-Door cars. Premium service should focus on the luxury segment.")

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x="Coverage", y="Customer Lifetime Value",
                    order=["Basic", "Extended", "Premium"], errorbar=None, color=ORANGE, ax=ax)
        ax.set_ylabel("Average CLV ($)")
        st.pyplot(fig)
        st.info("CLV rises with coverage level. Most customers are on Basic, "
                "so moving them up is an upselling chance.")

    with tab3:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Monthly Premium Auto", y="Customer Lifetime Value",
                        alpha=0.3, color=TEAL, ax=ax)
        st.pyplot(fig)
        st.info("Customers who pay a higher monthly premium tend to have a higher CLV.")

# ---------------------------------------------------------------- Page 3
else:
    hero("Predict Customer Lifetime Value", "Enter the customer details and click Predict")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            premium = st.number_input("💵 Monthly Premium Auto ($)", min_value=50, max_value=300, value=90)
            policies = st.slider("📄 Number of Policies", 1, 9, 2)
            income = st.number_input("💰 Annual Income ($)", min_value=0, max_value=150000, value=40000, step=1000)
            claim = st.number_input("🧾 Total Claim Amount ($)", min_value=0, max_value=3000, value=400)
        with col2:
            last_claim = st.slider("⏱️ Months Since Last Claim", 0, 35, 12)
            inception = st.slider("📅 Months Since Policy Inception", 0, 99, 24)
            coverage = st.selectbox("🛡️ Coverage", ["Basic", "Extended", "Premium"])
            vehicle = st.selectbox("🚙 Vehicle Class", sorted(df["Vehicle Class"].unique()))

        predict = st.button("🔮 Predict", type="primary", width="stretch")

    if predict:
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
        percentile = (df["Customer Lifetime Value"] < prediction).mean() * 100

        if percentile >= 80:
            level, css = "HIGH VALUE", "high"
            action = "Give retention offers, a dedicated agent and cross-sell extra policies or higher coverage."
        elif percentile >= 40:
            level, css = "MEDIUM VALUE", "medium"
            action = "Keep regular service. Offer an upgrade in coverage to grow the value."
        else:
            level, css = "LOW VALUE", "low"
            action = "Use low-cost, automated marketing (email, app messages). Avoid expensive retention offers."

        st.markdown(
            f'<div class="result {css}"><p>Predicted Customer Lifetime Value</p>'
            f'<h1>${prediction:,.0f}</h1><p>{level}</p></div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Compared with average", f"{prediction / avg:.1f}x", f"avg ${avg:,.0f}", delta_color="off")
        m2.metric("Higher than", f"{percentile:.0f}% of customers")
        m3.metric("Typical model error", "about $1,500")

        st.progress(int(percentile) / 100, text=f"Customer rank: top {100 - percentile:.0f}%")
        card("💡 Recommended action", action)

        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df["Customer Lifetime Value"], bins=50, color=TEAL, ax=ax)
        ax.axvline(prediction, color="red", linestyle="--", linewidth=2, label="This customer")
        ax.legend()
        ax.set_xlim(0, 40000)
        ax.set_xlabel("Customer Lifetime Value ($)")
        st.pyplot(fig)
