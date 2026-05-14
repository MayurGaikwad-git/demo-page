import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Process AI - Advanced Diagnostics", layout="wide")


# --- BACKEND: MODEL TRAINING & METRICS GENERATION ---
@st.cache_resource
def load_and_train_full():
    # Load Data
    df = pd.read_csv('main_data.csv')
    df['Target_Binary'] = df['Application_Status'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)

    # Process Cases
    case_targets = df.groupby('Case_ID')['Target_Binary'].max().reset_index()
    case_targets.rename(columns={'Target_Binary': 'Target'}, inplace=True)
    df_sequences = df.groupby('Case_ID')['Activity_Name'].apply(list).reset_index()
    df_merged = pd.merge(df_sequences, case_targets, on='Case_ID')

    all_acts = sorted(df['Activity_Name'].unique())

    # Feature Engineering
    def build_features(df_merged, all_acts):
        rows = []
        for _, row in df_merged.iterrows():
            seq = row['Activity_Name']
            for i in range(1, min(len(seq), 11)):
                sub_seq = seq[:i]
                feat = {'Case_ID': row['Case_ID'], 'Prefix_Length': i, 'Target': row['Target']}
                for p in range(10): feat[f'pos_{p + 1}'] = sub_seq[p] if p < len(sub_seq) else 'NONE'
                for act in all_acts: feat[f'freq_{act}'] = sub_seq.count(act)
                rows.append(feat)
        return pd.DataFrame(rows).fillna(0)

    feat_df = build_features(df_merged, all_acts)
    le = LabelEncoder()
    le.fit(all_acts + ['NONE'])
    for i in range(1, 11): feat_df[f'pos_{i}'] = le.transform(feat_df[f'pos_{i}'].astype(str))

    X = feat_df.drop(columns=['Case_ID', 'Target'])
    y = feat_df['Target']

    # Split for Validation Metrics
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=feat_df['Case_ID']))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    # Train Model
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Generate Validation Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, le, all_acts, X.columns, X_test, y_test, y_pred, y_prob


# Init Backend
model, le, all_acts, feature_cols, X_test, y_test, y_pred, y_prob = load_and_train_full()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Process AI")
page = st.sidebar.selectbox("Navigation", ["🔍 Prediction Dashboard", "📈 Detailed Model Insights"])

# --- PAGE 1: PREDICTION ---
if page == "🔍 Prediction Dashboard":
    st.title("Case Prediction for Acceptance ")
    uploaded_file = st.file_uploader("Upload single case event log (CSV)", type="csv")

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        trace = input_df['Activity_Name'].tolist()[:10]

        # Prepare Features
        feat = {'Prefix_Length': len(trace)}
        for i in range(10):
            val = trace[i] if i < len(trace) else 'NONE'
            feat[f'pos_{i + 1}'] = le.transform([val])[0] if val in le.classes_ else le.transform(['NONE'])[0]
        for act in all_acts: feat[f'freq_{act}'] = trace.count(act)

        # Inference
        df_inf = pd.DataFrame([feat]).reindex(columns=feature_cols, fill_value=0)
        prob = model.predict_proba(df_inf)[0][1]

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prob >= 0.5:
                st.success("### Prediction: ACCEPTED")
            else:
                st.error("### Prediction: REJECTED")
        with res_col2:
            st.metric("Probability Confidence", f"{prob * 100:.1f}%")
            st.progress(prob)

# --- PAGE 2: DETAILED MODEL INSIGHTS ---
elif page == "📈 Detailed Model Insights":
    st.title("Advanced Model Diagnostics")
    st.markdown("Yeh section model ki training performance aur decision-making process ko detail mein dikhata hai.")

    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    m4.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

    # Tabs for Details
    t1, t2, t3, t4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "🧬 Classification Report", "🏆 Feature Importance"])

    with t1:
        st.subheader("Confusion Matrix")
        st.write(
            "Yeh graph batata hai ki model ne kitne cases sahi predict kiye aur kahan 'False Positives' ya 'False Negatives' aaye.")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        st.pyplot(fig_cm)

    with t2:
        st.subheader("Receiver Operating Characteristic (ROC)")
        st.write("AUC curve model ki sensitivity aur specificity ke beech ka balance dikhata hai.")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_prob):.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    with t3:
        st.subheader("Full Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Greens'), use_container_width=True)

    with t4:
        st.subheader("Top Predictive Factors")
        st.write("Model in features ko sabse zyada importance deta hai final decision lene ke liye.")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
        fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax_fi)
        st.pyplot(fig_fi)

st.sidebar.markdown("---")
st.sidebar.write("**Model Config:** Random Forest | Balanced Classes | Prefix Window: 10")
