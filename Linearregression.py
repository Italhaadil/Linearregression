import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App
st.title("Linear Regression App")

# Sidebar - File upload
st.sidebar.header("Upload Your Dataset")
file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if file is not None:
    data = pd.read_csv(file)
    st.write("## Dataset Preview")
    st.write(data.head())

    # Select features and target
    st.sidebar.header("Select Features and Target")
    features = st.sidebar.multiselect("Select Independent Variables", data.columns.tolist())
    target = st.sidebar.selectbox("Select Dependent Variable", data.columns.tolist())

    if features and target:
        X = data[features]
        y = data[target]

        # Train-Test Split
        test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        random_state = st.sidebar.number_input("Random State", 0, 1000, 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("## Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        # Coefficients
        st.write("## Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        })
        st.write(coef_df)

        # Visualization
        if len(features) == 1:
            st.write("## Visualization")
            plt.figure(figsize=(8, 6))
            plt.scatter(X_test, y_test, color="blue", label="Actual")
            plt.plot(X_test, y_pred, color="red", label="Predicted")
            plt.xlabel(features[0])
            plt.ylabel(target)
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("Visualization is available only for single independent variable.")
    else:
        st.write("Please select independent variables and dependent variable from the sidebar.")
else:
    st.write("Please upload a CSV file to proceed.")
