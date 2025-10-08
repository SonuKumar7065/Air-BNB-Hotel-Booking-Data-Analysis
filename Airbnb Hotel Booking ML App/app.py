# ============================================
# 🏨 Airbnb Hotel Booking Data Analysis & Price Prediction App
# Author: Sonu Kumar
# Internship: VOIS & Vodafone Idea Foundation – Conversational Data Analytics with LLMs
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Airbnb Data Analysis & Price Prediction",
    page_icon="🏨",
    layout="wide"
)
st.title("🏨 Airbnb Hotel Booking Data Analysis & Price Prediction")
st.markdown("#### Created by **Sonu Kumar** | VOIS for Tech Internship")

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    df = pd.read_excel("Airbnb_Open_Data.xlsx")
    df = df.drop_duplicates()
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'Airbnb_Open_Data.xlsx' not found. Please place it in the same folder as this app.")
    st.stop()

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("🔍 Filter Options")

city_filter = st.sidebar.multiselect(
    "Select City", options=sorted(df['city'].dropna().unique())
)
if city_filter:
    df = df[df['city'].isin(city_filter)]

property_type_filter = st.sidebar.multiselect(
    "Select Property Type", options=sorted(df['property_type'].dropna().unique())
)
if property_type_filter:
    df = df[df['property_type'].isin(property_type_filter)]

# ============================================
# MAIN TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🤖 ML Model", "💡 Insights"])

# ============================================
# TAB 1 - DATA ANALYSIS
# ============================================
with tab1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Listings", len(df))
    c2.metric("Average Price ($)", round(df['price'].mean(), 2))
    c3.metric("Median Price ($)", round(df['price'].median(), 2))

    st.write("### 💰 Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['price'], bins=50, kde=True, ax=ax, color="teal")
    ax.set_title("Price Distribution of Airbnb Listings")
    st.pyplot(fig)

    st.write("### 🌆 Average Price by Room Type")
    if 'room_type' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df.groupby('room_type')['price'].mean().sort_values(ascending=False).plot(
            kind='bar', color='coral', ax=ax
        )
        plt.xticks(rotation=45)
        ax.set_ylabel("Average Price ($)")
        st.pyplot(fig)

    st.write("### 🔥 Correlation Heatmap (Numerical Features)")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ============================================
# TAB 2 - MACHINE LEARNING MODEL
# ============================================
with tab2:
    st.subheader("🤖 Price Prediction Using Random Forest")

    # Prepare dataset
    df_model = df.select_dtypes(include=[np.number]).dropna()

    if 'price' not in df_model.columns:
        st.error("Dataset doesn't contain 'price' column for model training.")
    else:
        X = df_model.drop(columns=['price'])
        y = df_model['price']

        if X.empty or y.empty:
            st.error("Not enough numerical data for model training. Try removing filters.")
        else:
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Evaluation
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.write("### 📈 Model Evaluation Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.3f}")

            # Feature Importance (fixed)
            try:
                importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.write("### 🔍 Top 10 Important Features")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    x=importances.values[:10], 
                    y=importances.index[:10], 
                    palette="viridis", ax=ax
                )
                ax.set_title("Feature Importance (Random Forest)")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Could not compute feature importances: {e}")

            # Prediction section
            st.markdown("---")
            st.subheader("💡 Try Your Own Prediction")

            user_input = {}
            st.write("Enter sample values below:")
            for col in X.columns[:5]:
                val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
                user_input[col] = val

            if st.button("Predict Price 💰"):
                try:
                    user_df = pd.DataFrame([user_input])
                    user_scaled = scaler.transform(user_df)
                    pred_price = model.predict(user_scaled)[0]
                    st.success(f"🏠 Predicted Price: **${pred_price:.2f}**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# ============================================
# TAB 3 - INSIGHTS
# ============================================
with tab3:
    st.subheader("💡 Key Insights & Findings")

    st.markdown("""
    ✅ Most Airbnb listings are concentrated in major tourist cities.  
    ✅ **Room type** and **reviews** have a strong impact on pricing.  
    ✅ The **Random Forest** model achieves reliable accuracy for predicting prices.  
    ✅ Data cleaning significantly improves the model’s performance.  
    ✅ The dashboard helps visualize trends and guide pricing decisions.
    """)

    st.markdown("#### 📌 Developed by **Sonu Kumar** | 2025 Internship Project")
