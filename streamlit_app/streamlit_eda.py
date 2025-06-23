import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Set Streamlit page config
st.set_page_config(
    page_title="Mobile Money EDA", 
    layout="wide",
    page_icon="📱"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH", "data/transactions.csv")

# Load dataset with caching
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATASET_PATH)

# Fix Arrow serialization issues
object_cols = df.select_dtypes(include='object').columns
df[object_cols] = df[object_cols].astype(str)

# Header
st.markdown('<h1 class="main-header">📱 Mobile Money Transaction EDA</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Exploratory Data Analysis of Mobile Money Transaction Patterns</p>', unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")

with col2:
    st.metric("Total Volume", f"${df['amount'].sum():,.0f}")

with col3:
    fraud_rate = (df['isFraud'].sum() / len(df)) * 100
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

with col4:
    st.metric("Transaction Types", df['transactionType'].nunique())

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Data Overview Section
st.header("🔍 Dataset Overview")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("Dataset Info")
    memory = df.memory_usage(deep=True).sum() / 1024**2
    st.info(f"""
    **Shape:** {df.shape[0]:,} × {df.shape[1]}  
    **Memory:** {memory:.2f} MB  
    **Null Values:** {df.isnull().sum().sum()}
    """)

st.subheader("Summary Statistics")
st.dataframe(df.describe(include='all'), use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Transaction Analysis
st.header("📊 Transaction Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Types")
    trans_type_counts = df['transactionType'].value_counts()
    fig = px.bar(x=trans_type_counts.index, y=trans_type_counts.values, 
                 color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title="Transaction Type", yaxis_title="Count", 
                      showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Fraud Distribution")
    fraud_counts = df['isFraud'].value_counts()
    fraud_labels = ['Legitimate', 'Fraudulent']
    colors = ['#2ca02c', '#d62728']
    fig = px.pie(values=fraud_counts.values, names=fraud_labels, 
                 color_discrete_sequence=colors)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Transaction Types by Fraud Count")
fraud_by_type = df[df['isFraud'] == 1]['transactionType'].value_counts().reset_index()
fraud_by_type.columns = ['transactionType', 'FraudCount']
fig = px.bar(fraud_by_type, x='transactionType', y='FraudCount', color_discrete_sequence=['#d62728'])
fig.update_layout(xaxis_title="Transaction Type", yaxis_title="Fraud Count", height=400)
st.plotly_chart(fig, use_container_width=True)

# Amount Analysis
st.subheader("⏳ Transaction Volume Over Steps")
volume_by_step = df.groupby('step')['amount'].sum().reset_index()
fig = px.line(volume_by_step, x='step', y='amount', title="Transaction Volume Over Time", markers=True)
fig.update_layout(xaxis_title="Step", yaxis_title="Total Amount", height=400)
st.plotly_chart(fig, use_container_width=True)

st.subheader("💰 Transaction Amount Distribution")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x='amount', nbins=50, color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title="Amount", yaxis_title="Frequency", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df, x='transactionType', y='amount')
    fig.update_layout(xaxis_title="Transaction Type", yaxis_title="Amount", height=400)
    st.plotly_chart(fig, use_container_width=True)

# st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("📈 Correlation Heatmap")
numeric_df = df.select_dtypes(include=['int', 'float'])
corr = numeric_df.corr(numeric_only=True)
fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)


# Interactive Filter Section
st.header("🎛️ Interactive Filter")

col1, col2 = st.columns([1, 3])

with col1:
    transaction_options = df['transactionType'].unique().tolist()
    selected_type = st.selectbox("Select Transaction Type", transaction_options)
    
    show_fraud_only = st.checkbox("Show fraudulent transactions only")

with col2:
    filtered_df = df[df['transactionType'] == selected_type]
    
    if show_fraud_only:
        filtered_df = filtered_df[filtered_df['isFraud'] == 1]
    
    st.write(f"**Showing {len(filtered_df):,} transactions for: {selected_type}**")
    
    if show_fraud_only and len(filtered_df) == 0:
        st.info("No fraudulent transactions found for this transaction type.")
    else:
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # Quick stats for filtered data
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Count", f"{len(filtered_df):,}")
        with col_b:
            st.metric("Total Amount", f"${filtered_df['amount'].sum():,.0f}")
        with col_c:
            avg_amount = filtered_df['amount'].mean() if len(filtered_df) > 0 else 0
            st.metric("Average Amount", f"${avg_amount:,.2f}")