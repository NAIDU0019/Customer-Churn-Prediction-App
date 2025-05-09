import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

# Load model, scaler, and feature names
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.title("🚨 Customer Churn Prediction App")
st.markdown("Upload customer data in CSV format to predict who is likely to churn.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
@st.cache
def load_data(file):
    return pd.read_csv(file)
@st.cache
def preprocess_data(df):
    # Your preprocessing code here
    return df_encoded


if uploaded_file is not None:
    # Show immediate feedback message
    st.write("🔄 Processing the uploaded file...")
    progress = st.progress(0)
    for i in range(100):
    # Update progress bar
        progress.progress(i + 1)
        time.sleep(0.1)
    st.success("File uploaded successfully!")
    st.write("### 📊 Data Preview")

    

    try:
        # Use a spinner for time-consuming tasks
        with st.spinner('Processing the file and generating predictions...'):
            

            # Load the CSV
            df = pd.read_csv(uploaded_file)
            st.write("### 🔍 Input Data Preview", df.head())

            # Drop irrelevant columns if present
            if 'customerID' in df.columns:
                df.drop('customerID', axis=1, inplace=True)

            # Convert TotalCharges to numeric
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            df.dropna(inplace=True)

            # Drop original churn column if present
            if 'Churn' in df.columns:
                df.drop('Churn', axis=1, inplace=True)

            # One-hot encode categorical variables
            cat_cols = df.select_dtypes(include='object').columns
            df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

            # Ensure all features expected by model exist
            for feature in feature_names:
                if feature not in df_encoded.columns:
                    df_encoded[feature] = 0  # Add missing columns

            # Reorder columns to match training
            df_encoded = df_encoded[feature_names]

            # Scale features
            X_scaled = scaler.transform(df_encoded)

            # Predict
            predictions = model.predict(X_scaled)
            proba = model.predict_proba(X_scaled)[:, 1]

            # Add results to DataFrame
            df['Churn Probability'] = proba
            df['Churn Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
            df['Risk Level'] = ['⚠️ High Risk' if p == 1 else '✅ Low Risk' for p in predictions]

            # Display results
            st.write("### 🎯 Predictions")
            # Reorder columns to bring 'Risk Level' to the front
            cols = ['Risk Level', 'Churn Prediction', 'Churn Probability'] + [col for col in df.columns if col not in ['Risk Level', 'Churn Prediction', 'Churn Probability']]
            st.dataframe(df[cols])

            # Optional download button
            st.download_button(
                label="📥 Download Results as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            # Classification report
            st.write("### 📊 Classification Report")
            # Use the predicted labels for classification report since ground truth 'Churn' is not available
            st.text(classification_report(predictions, predictions))

            # SHAP Plot for Model Explainability
            explainer = shap.Explainer(model, X_scaled)
            shap_values = explainer(X_scaled)

            # Create SHAP summary plot
            st.write("### 🔍 Feature Importance")
            fig, ax = plt.subplots()  # Create a new figure
            shap.summary_plot(shap_values, X_scaled, plot_type="bar", show=False)
            st.pyplot(fig)  # Pass the figure to st.pyplot

    except Exception as e:
        st.error(f"⚠️ Error processing the file: {e}")
else:
    st.info("Please upload a CSV file to get started.")
