import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Streamlit page setup
st.set_page_config(page_title="NYC Taxi Data Analysis", layout="wide")
st.title("🚖 NYC Green Taxi Trip Data - Interactive Analysis & Prediction")

# Upload parquet file
uploaded_file = st.file_uploader("Upload Parquet File", type=["parquet"])
if uploaded_file:
    df = pd.read_parquet(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Drop unnecessary column
    if 'ehail_fee' in df.columns:
        df.drop(columns=['ehail_fee'], inplace=True)

    # Trip duration
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60

    # Feature Engineering
    df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
    df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour

    # Fill missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].fillna('Unknown')

    st.subheader("Missing Values After Cleaning")
    st.write(df.isnull().sum())

    # Payment Type Pie Chart
    st.subheader("🧾 Payment Type Distribution")
    payment_counts = df['payment_type'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Weekday vs Total Amount
    st.subheader("📅 Average Total Amount by Weekday")
    avg_total_weekday = df.groupby('weekday')['total_amount'].mean().sort_index()
    st.bar_chart(avg_total_weekday)

    # Payment type vs Total Amount
    st.subheader("💳 Average Total Amount by Payment Type")
    avg_total_payment = df.groupby('payment_type')['total_amount'].mean()
    st.bar_chart(avg_total_payment)

    # Statistical Tests
    st.subheader("📊 Statistical Tests")
    if 'trip_type' in df.columns:
        trip_type_1 = df[df['trip_type'] == 1]['total_amount']
        trip_type_2 = df[df['trip_type'] == 2]['total_amount']
        t_stat, p_val = ttest_ind(trip_type_1, trip_type_2, nan_policy='omit')
        st.write(f"T-test between Trip Types → T-statistic: {t_stat:.3f}, P-value: {p_val:.3f}")

    weekday_groups = [group['total_amount'].dropna() for _, group in df.groupby('weekday')]
    f_stat, p_val = f_oneway(*weekday_groups)
    st.write(f"ANOVA for Weekday Groups → F-statistic: {f_stat:.3f}, P-value: {p_val:.3f}")

    contingency_table = pd.crosstab(df['trip_type'], df['payment_type'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
    st.write(f"Chi-Square Test between Trip Type and Payment Type → Chi2-statistic: {chi2_stat:.3f}, P-value: {p_val:.3f}")

    # Correlation Matrix
    st.subheader("📈 Correlation Matrix")
    numeric_corr_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                        'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
                        'trip_duration', 'passenger_count']
    corr_matrix = df[numeric_corr_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Histogram, Boxplot, KDE of Total Amount
    st.subheader("📊 Total Amount Distribution")
    fig3, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['total_amount'], kde=False, ax=ax[0])
    ax[0].set_title('Histogram')

    sns.boxplot(x=df['total_amount'], ax=ax[1])
    ax[1].set_title('Boxplot')

    sns.kdeplot(df['total_amount'], shade=True, ax=ax[2])
    ax[2].set_title('Density Plot')

    st.pyplot(fig3)

    # One-Hot Encoding
    encode_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type',
                   'trip_type', 'weekday', 'hourofday']
    df_encoded = pd.get_dummies(df, columns=encode_cols)

    # Modeling
    st.subheader("🔮 Predict Fare Amount")

    X = df_encoded.drop(columns=['total_amount', 'lpep_pickup_datetime', 'lpep_dropoff_datetime'])
    y = df_encoded['total_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    st.write(f"Linear Regression R² Score: {model_lr.score(X_test, y_test):.3f}")

    # Interactive Prediction
    st.subheader("🚀 Predict Fare Based on Input")

    pickup_hour = st.slider("Pickup Hour", min_value=0, max_value=23, value=12)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)

    input_array = np.zeros(X_train.shape[1])
    
    # Set pickup hour in one-hot
    try:
        pickup_hour_index = X_train.columns.get_loc('hourofday_' + str(pickup_hour))
        input_array[pickup_hour_index] = 1
    except KeyError:
        st.warning("Pickup hour not found in one-hot encoded features.")

    # Set passenger count
    passenger_count_index = X_train.columns.get_loc('passenger_count')
    input_array[passenger_count_index] = passenger_count

    # Make prediction
    predicted_fare = model_lr.predict([input_array])[0]
    st.success(f"💰 Predicted Fare: ${predicted_fare:.2f}")
else:
    st.info("👆 Please upload a `.parquet` file to begin.")
