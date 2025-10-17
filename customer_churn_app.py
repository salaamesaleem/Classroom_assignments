# customer_churn_app.py
# Single-file pipeline + Streamlit app for Customer Churn prediction.
# - Sections marked with big comment blocks (SEARCH for "STEP X" to jump).
# - Saves preprocessor and top models to ./churn_models/
# - UI allows single-row input and CSV batch upload, plus a "Retrain models" button.

# ===== DEPENDENCIES =====
# pip install pandas scikit-learn joblib streamlit

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# sklearn imports used for training and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ===== CONFIG =====
DATA_FILE = "customer_churn_dataset-training-master.csv"   # expected dataset filename
MODELS_DIR = Path("./churn_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"

# ===== HELPER FUNCTIONS =====
@st.cache_data
def load_preprocessor():
    if PREPROCESSOR_FILE.exists():
        return joblib.load(PREPROCESSOR_FILE)
    return None

@st.cache_data
def load_models():
    models = {}
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        if p.name == PREPROCESSOR_FILE.name:
            continue
        models[p.stem] = joblib.load(p)
    return models

def save_preprocessor_and_models(preprocessor, models_dict, metadata):
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    for name, mdl in models_dict.items():
        joblib.dump(mdl, MODELS_DIR / f"{name}.joblib")
    # save metadata as JSON
    pd.Series(metadata).to_json(METADATA_FILE)

def build_preprocessor(X_train):
    # numeric and categorical columns lists
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
    ], remainder='drop')
    return preprocessor, numeric_cols, categorical_cols

def train_and_save_default_models(sample_rows=None):
    """Train simple baseline models and save preprocessor + models.
       sample_rows: optionally limit dataset size (int) for faster training
    """
    df = pd.read_csv(DATA_FILE)
    # quick cleaning: drop rows that are completely empty
    df = df.dropna(how='all')

    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # ensure target int
    df[target_col] = df[target_col].astype(int)

    # optional sampling for speed
    if sample_rows is not None and len(df) > sample_rows:
        # stratified sample to preserve class distribution
        df = df.groupby(target_col, group_keys=False).apply(lambda x: x.sample(min(len(x), sample_rows//2), random_state=42))
        df = df.sample(frac=1, random_state=42)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # build preprocessor
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # transform
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # models to train
    models_to_train = {
        "logistic": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        # "hist_gb": HistGradientBoostingClassifier(random_state=42)  # optional: uncomment to include
    }

    results = {}
    for name, mdl in models_to_train.items():
        mdl.fit(X_train_t, y_train)
        preds = mdl.predict(X_test_t)
        probs = mdl.predict_proba(X_test_t)[:,1] if hasattr(mdl, "predict_proba") else None
        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs) if probs is not None else None
        results[name] = {"model": mdl, "accuracy": acc, "roc_auc": roc}
        st.write(f"Trained {name}: accuracy={acc:.4f}" + (f", roc_auc={roc:.4f}" if roc is not None else ""))

    # prepare metadata and save
    metadata = {
        "features": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "model_ranking": [{ "name": n, "accuracy": results[n]['accuracy'], "roc_auc": results[n]['roc_auc']} for n in results]
    }
    # Save preprocessor and models
    save_preprocessor_and_models(preprocessor, {k:v['model'] for k,v in results.items()}, metadata)
    return results, metadata

def predict_input(model, preprocessor, input_df):
    X_t = preprocessor.transform(input_df)
    pred = model.predict(X_t)[0]
    prob = model.predict_proba(X_t)[0,1] if hasattr(model, "predict_proba") else None
    return int(pred), float(prob) if prob is not None else None

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Predictor — single-file app")
st.markdown("This app loads saved models (in `./churn_models/`) or allows retraining. Use the sidebar to control models.")

# Sidebar controls
st.sidebar.header("Controls")
do_retrain = st.sidebar.button("Retrain models (overwrite saved models)")
sample_for_retrain = st.sidebar.number_input("Sample rows for retraining (0 = use full dataset)", value=30000, min_value=0, step=1000)
available_models = list(load_models().keys())
selected_model_key = st.sidebar.selectbox("Select model to use (saved)", options=available_models if available_models else ["(no saved model)"])

# Retrain if requested
if do_retrain:
    st.sidebar.info("Retraining started. See logs in main panel.")
    try:
        sample_rows = int(sample_for_retrain) if sample_for_retrain > 0 else None
        results, metadata = train_and_save_default_models(sample_rows=sample_rows)
        st.success("Retraining finished and saved in ./churn_models/")
        available_models = list(load_models().keys())
        selected_model_key = available_models[0] if available_models else selected_model_key
    except Exception as e:
        st.error(f"Retraining failed: {e}")

# Load preprocessor and models
preprocessor = load_preprocessor()
models = load_models()

# Single-row input form
st.header("Predict churn for a single customer")
if preprocessor is None:
    st.warning("No preprocessor found. Retrain models first using the sidebar button.")
else:
    meta = None
    if Path(METADATA_FILE).exists():
        meta = pd.read_json(METADATA_FILE, typ='series')
    numeric_cols = list(meta['numeric_cols']) if meta is not None else []
    categorical_cols = list(meta['categorical_cols']) if meta is not None else []

    with st.form("single_input"):
        st.subheader("Enter customer features")
        input_vals = {}
        # numeric inputs
        for c in numeric_cols:
            # pick a default based on column name (0) - user will edit
            input_vals[c] = st.number_input(f"{c}", value=0.0, format="%.4f")
        # categorical inputs
        for c in categorical_cols:
            input_vals[c] = st.text_input(f"{c}", value="")
        submitted = st.form_submit_button("Predict")
    if submitted:
        input_df = pd.DataFrame([input_vals])
        # ensure all preprocessor inputs exist
        # (the preprocessor was fit with a specific list; scikit-learn ColumnTransformer sometimes has attribute feature_names_in_)
        try:
            # If preprocessor has feature_names_in_ (sklearn >=1.0), use it; else we assume columns match by name.
            feature_names = preprocessor.feature_names_in_ if hasattr(preprocessor, "feature_names_in_") else input_df.columns
        except Exception:
            feature_names = input_df.columns
        # add missing numeric/categorical columns
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0 if col in numeric_cols else ""
        input_df = input_df[feature_names]

        if not models:
            st.error("No saved models available. Retrain first.")
        else:
            # If the selected model isn't present fallback to first saved model
            if selected_model_key not in models:
                selected_model = list(models.values())[0]
            else:
                selected_model = models[selected_model_key]
            pred, prob = predict_input(selected_model, preprocessor, input_df)
            st.write("**Prediction:**", ("Churn (1)" if pred==1 else "No churn (0)"))
            if prob is not None:
                st.write(f"Predicted churn probability: {prob:.3f}")
            st.write("Input used:")
            st.json(input_vals)

# Batch prediction
st.header("Batch prediction (CSV upload)")
uploaded = st.file_uploader("Upload CSV with same feature columns used for training", type=["csv"])
if uploaded:
    df_upload = pd.read_csv(uploaded)
    if preprocessor is None:
        st.error("No preprocessor available. Retrain first.")
    else:
        # transform and predict with first available model
        if not models:
            st.error("No saved models. Retrain first.")
        else:
            mdl = list(models.values())[0]
            X_t = preprocessor.transform(df_upload)
            preds = mdl.predict(X_t)
            probs = mdl.predict_proba(X_t)[:,1] if hasattr(mdl, "predict_proba") else None
            df_upload['pred_churn'] = preds
            if probs is not None:
                df_upload['pred_churn_proba'] = probs
            st.dataframe(df_upload.head(50))
            st.download_button("Download predictions (CSV)", df_upload.to_csv(index=False).encode('utf-8'), "predictions.csv")

# Show metadata/models info
st.header("Saved models & metadata")
if Path(METADATA_FILE).exists():
    st.write("Metadata (training-time):")
    st.json(pd.read_json(METADATA_FILE, typ='series').to_dict())
else:
    st.write("No metadata found (no previous training run). Use Retrain to create models and metadata.")
st.write("Saved model files:")
for p in sorted(MODELS_DIR.glob("*.joblib")):
    st.write("-", p.name)

st.write("End of single-file churn app.")
