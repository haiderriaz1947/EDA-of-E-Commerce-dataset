
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io

st.set_page_config(page_title="Ecommerce EDA", layout="wide")
st.title("Ecommerce Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload Ecommerce Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load Data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)
    st.write("Shape of dataset:", df.shape)

    # ---- Basic Info ----
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("Missing Values:", df.isnull().sum().to_dict())
    st.write("Duplicate Records:", df.duplicated().sum())

    # ---- Data Cleaning ----
    if "price" in df.columns:
        df["price"] = df["price"].astype(float)

    # Derived Column: Sales
    if {"price", "quantity", "discount"}.issubset(df.columns):
        df["sales"] = df["price"] * df["quantity"] * (1 - df["discount"])
        df["sales"] = df["sales"].astype(float)

    # Convert date
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["day_name"] = df["order_date"].dt.day_name()

    st.subheader("Cleaned Data Sample")
    st.dataframe(df.head())

    # ---- Category Analysis ----
    st.header("Category Analysis")

    if "category" in df.columns:
        st.subheader("Category Distribution")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.bar(cat_counts, x="category", y="count",
                     title="Category Counts", text="count", color="category")
        st.plotly_chart(fig, use_container_width=True)

    if {"category", "sales"}.issubset(df.columns):
        st.subheader("Total Sales by Category")
        cat_sales = df.groupby("category")["sales"].sum().reset_index()
        fig = px.bar(cat_sales, x="category", y="sales", title="Sales by Category", color="category")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Region-wise Sales ----
    if {"region", "sales"}.issubset(df.columns):
        st.header("Regional Analysis")
        st.subheader("Sales Share by Region")
        fig = px.pie(df, names="region", values="sales", title="Sales Contribution by Region")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Product Analysis ----
    st.header("Product Analysis")

    if {"product_id", "sales"}.issubset(df.columns):
        st.subheader("Top 10 Best-Selling Products (by Sales)")
        top_products = df.groupby("product_id")["sales"].sum().nlargest(10).reset_index()
        fig = px.bar(top_products, x="product_id", y="sales", title="Top 10 Products by Sales", color="sales")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Lowest Performing Products")
        low_products = df.groupby("product_id")["sales"].sum().nsmallest(10).reset_index()
        fig = px.bar(low_products, x="product_id", y="sales", title="Lowest Performing Products", color="sales")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Customer Analysis ----
    st.header("Customer Analysis")

    if {"customer_id", "sales"}.issubset(df.columns):
        st.subheader("Top 10 Customers by Sales")
        cust_sales = df.groupby("customer_id")["sales"].sum().reset_index()
        cust_sales = cust_sales.sort_values("sales", ascending=False).head(10)
        fig = px.bar(cust_sales, x="customer_id", y="sales", title="Top 10 Customers by Sales", color="sales")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Weekday Sales ----
    if "day_name" in df.columns:
        st.header("Time-based Analysis")
        st.subheader("Sales by Day of Week")
        weekday_sales = df.groupby("day_name")["sales"].sum().reset_index()
        fig = px.bar(weekday_sales, x="day_name", y="sales", color="day_name", title="Total Sales by Weekday")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Correlation ----
    st.header("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        corr = numeric_cols.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, cmap="coolwarm", annot=True)
        st.pyplot(fig)

else:
    st.warning("Please upload a dataset file to proceed.")
