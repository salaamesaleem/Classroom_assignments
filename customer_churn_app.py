# ==========================================================
# 📘 Customer Churn Predictor (Single File - Streamlit Cloud)
# Author: Saleem
# ==========================================================
# ▶ How to Deploy:
# 1️⃣ Upload this ONE FILE to a new GitHub repo.
# 2️⃣ Go to https://share.streamlit.io → "New App"
# 3️⃣ Choose your repo and file → Deploy 🚀
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================================
# STEP 1: Title & Setup
# ==========================================================
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Predictor — Single File Deployment")
st.write("Upload a dataset, train churn models, and make predictions easily!")

# ==========================================================
# STEP 2: Upload Dataset
# ==========================================================
st.header("Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV file containing customer churn data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Preview of your data:")
    st.dataframe(df.head())
else:
    st.info("👆 Please upload a CSV file to continue.")
    st.stop()

# ==========================================================
# STEP 3: Select Target Column
# ==========================================================
st.header("Step 2: Select Target Column")
target_col = st.selectbox("Select the target column (e.g., Churn):", options=df.columns)

if not target_col:
    st.warning("Please select a target column to continue.")
    st.stop()

# ==========================================================
# STEP 4: Train Models
# ==========================================================
st.header("Step 3: Train Models")

if st.button("🚀 Train Models"):
    try:
        df = df.dropna(how='all')
        if 'CustomerID' in df.columns:
            df = df.drop(columns=['CustomerID'])

        df[target_col] = df[target_col].astype(int)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Preprocessing
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
        }

        # Train and evaluate
        model_results = {}
        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            acc = accuracy_score(y_test, preds)
            roc = roc_auc_score(y_test, probs) if probs is not None else None
            model_results[name] = (pipeline, acc, roc)

        # Display results
        st.subheader("📈 Model Performance")
        for name, (pipe, acc, roc) in model_results.items():
            st.write(f"**{name}** → Accuracy: `{acc:.3f}` | ROC-AUC: `{roc:.3f}`")

        best_model_name = max(model_results, key=lambda k: model_results[k][1])
        best_model = model_results[best_model_name][0]
        st.success(f"✅ Best model: {best_model_name}")

        st.session_state['best_model'] = best_model
        st.session_state['preprocessor'] = preprocessor
        st.session_state['columns'] = X.columns.tolist()
        st.session_state['numeric_cols'] = numeric_cols
        st.session_state['categorical_cols'] = categorical_cols

    except Exception as e:
        st.error(f"Training failed: {e}")

# ==========================================================
# STEP 5: Single Prediction
# ==========================================================
if 'best_model' in st.session_state:
    st.header("Step 4: Predict for a Single Customer")
    input_vals = {}

    for col in st.session_state['numeric_cols']:
        input_vals[col] = st.number_input(f"{col}", value=0.0)

    for col in st.session_state['categorical_cols']:
        input_vals[col] = st.text_input(f"{col}", value="")

    if st.button("🔮 Predict Churn"):
        input_df = pd.DataFrame([input_vals])
        model = st.session_state['best_model']
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0, 1] if hasattr(model, "predict_proba") else None

        st.write("### ✅ Prediction Result")
        st.write("**Churn:**", "Yes (1)" if pred == 1 else "No (0)")
        if prob is not None:
            st.write(f"**Probability:** {prob:.3f}")
        st.json(input_vals)

# ==========================================================
# STEP 6: Batch Prediction
# ==========================================================
if 'best_model' in st.session_state:
    st.header("Step 5: Batch Prediction (Upload CSV)")
    uploaded_batch = st.file_uploader("Upload CSV file for batch predictions", type=["csv"], key="batch_upload")

    if uploaded_batch:
        df_upload = pd.read_csv(uploaded_batch)
        model = st.session_state['best_model']

        preds = model.predict(df_upload)
        probs = model.predict_proba(df_upload)[:, 1] if hasattr(model, "predict_proba") else None

        df_upload['pred_churn'] = preds
        if probs is not None:
            df_upload['pred_churn_proba'] = probs

        st.dataframe(df_upload.head())
        st.download_button("📥 Download Predictions", df_upload.to_csv(index=False).encode('utf-8'), "predictions.csv")

st.write("💡 *End of single-file Streamlit Churn App*")
