# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.preprocessing import LabelEncoder
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import accuracy_score, classification_report
# # # from xgboost import XGBClassifier
# # # import plotly.express as px
# # #
# # # st.set_page_config(page_title="Predictive Process Monitoring", layout="wide")
# # #
# # # st.title("🚀 Insurance Claim Predictive Monitoring Dashboard")
# # #
# # # # =========================================================
# # # # FILE UPLOAD
# # # # =========================================================
# # #
# # # uploaded_file = st.file_uploader(
# # #     "Upload Insurance Event Log CSV",
# # #     type=["csv"]
# # # )
# # #
# # # if uploaded_file is not None:
# # #
# # #     df = pd.read_csv(uploaded_file)
# # #
# # #     st.subheader("📌 Raw Dataset")
# # #     st.dataframe(df.head())
# # #
# # #     # =========================================================
# # #     # COLUMN SELECTION
# # #     # =========================================================
# # #
# # #     st.sidebar.header("Column Mapping")
# # #
# # #     case_col = st.sidebar.selectbox(
# # #         "Case/Application ID",
# # #         df.columns
# # #     )
# # #
# # #     activity_col = st.sidebar.selectbox(
# # #         "Activity Column",
# # #         df.columns
# # #     )
# # #
# # #     timestamp_col = st.sidebar.selectbox(
# # #         "Timestamp Column",
# # #         df.columns
# # #     )
# # #
# # #     outcome_col = st.sidebar.selectbox(
# # #         "Final Outcome Column",
# # #         df.columns
# # #     )
# # #
# # #     # =========================================================
# # #     # PREPROCESSING
# # #     # =========================================================
# # #
# # #     df[timestamp_col] = pd.to_datetime(df[timestamp_col])
# # #
# # #     df = df.sort_values([case_col, timestamp_col])
# # #
# # #     # =========================================================
# # #     # TARGET VARIABLE
# # #     # =========================================================
# # #
# # #     # CHANGE THESE VALUES BASED ON YOUR DATASET
# # #     rejection_keywords = ["Rejected", "Reject", "Denied"]
# # #
# # #     df["target"] = df[outcome_col].astype(str).apply(
# # #         lambda x: 1 if any(k.lower() in x.lower() for k in rejection_keywords) else 0
# # #     )
# # #
# # #     # =========================================================
# # #     # FEATURE ENGINEERING
# # #     # =========================================================
# # #
# # #     feature_rows = []
# # #
# # #     grouped = df.groupby(case_col)
# # #
# # #     for case_id, group in grouped:
# # #
# # #         group = group.sort_values(timestamp_col)
# # #
# # #         activities = list(group[activity_col].astype(str))
# # #
# # #         first_activity = activities[0]
# # #
# # #         last_activity = activities[-1]
# # #
# # #         num_activities = len(activities)
# # #
# # #         unique_activities = len(set(activities))
# # #
# # #         rework_count = num_activities - unique_activities
# # #
# # #         duration = (
# # #             group[timestamp_col].max() -
# # #             group[timestamp_col].min()
# # #         ).total_seconds() / 3600
# # #
# # #         variant = " -> ".join(activities[:5])
# # #
# # #         target = group["target"].max()
# # #
# # #         row = {
# # #             "case_id": case_id,
# # #             "first_activity": first_activity,
# # #             "last_activity": last_activity,
# # #             "num_activities": num_activities,
# # #             "unique_activities": unique_activities,
# # #             "rework_count": rework_count,
# # #             "duration_hours": duration,
# # #             "variant": variant,
# # #             "target": target
# # #         }
# # #
# # #         feature_rows.append(row)
# # #
# # #     feature_df = pd.DataFrame(feature_rows)
# # #
# # #     st.subheader("📊 Engineered Features")
# # #     st.dataframe(feature_df.head())
# # #
# # #     # =========================================================
# # #     # ENCODING
# # #     # =========================================================
# # #
# # #     encoders = {}
# # #
# # #     categorical_cols = [
# # #         "first_activity",
# # #         "last_activity",
# # #         "variant"
# # #     ]
# # #
# # #     for col in categorical_cols:
# # #         le = LabelEncoder()
# # #         feature_df[col] = le.fit_transform(feature_df[col].astype(str))
# # #         encoders[col] = le
# # #
# # #     # =========================================================
# # #     # TRAIN TEST SPLIT
# # #     # =========================================================
# # #
# # #     X = feature_df.drop(columns=["case_id", "target"])
# # #
# # #     y = feature_df["target"]
# # #
# # #     X_train, X_test, y_train, y_test = train_test_split(
# # #         X,
# # #         y,
# # #         test_size=0.2,
# # #         random_state=42
# # #     )
# # #
# # #     # =========================================================
# # #     # MODEL TRAINING
# # #     # =========================================================
# # #
# # #     model = XGBClassifier(
# # #         n_estimators=100,
# # #         max_depth=5,
# # #         learning_rate=0.1,
# # #         eval_metric='logloss'
# # #     )
# # #
# # #     model.fit(X_train, y_train)
# # #
# # #     # =========================================================
# # #     # EVALUATION
# # #     # =========================================================
# # #
# # #     predictions = model.predict(X_test)
# # #
# # #     accuracy = accuracy_score(y_test, predictions)
# # #
# # #     st.subheader("📈 Model Accuracy")
# # #
# # #     st.metric("Accuracy", f"{accuracy:.2f}")
# # #
# # #     st.text(classification_report(y_test, predictions))
# # #
# # #     # =========================================================
# # #     # FEATURE IMPORTANCE
# # #     # =========================================================
# # #
# # #     importance_df = pd.DataFrame({
# # #         "Feature": X.columns,
# # #         "Importance": model.feature_importances_
# # #     }).sort_values("Importance", ascending=False)
# # #
# # #     fig = px.bar(
# # #         importance_df,
# # #         x="Importance",
# # #         y="Feature",
# # #         orientation="h",
# # #         title="Feature Importance"
# # #     )
# # #
# # #     st.plotly_chart(fig, use_container_width=True)
# # #
# # #     # =========================================================
# # #     # CASE PREDICTION
# # #     # =========================================================
# # #
# # #     st.subheader("🔍 Predict Individual Application")
# # #
# # #     selected_case = st.selectbox(
# # #         "Select Application ID",
# # #         feature_df["case_id"]
# # #     )
# # #
# # #     selected_row = feature_df[
# # #         feature_df["case_id"] == selected_case
# # #     ]
# # #
# # #     input_data = selected_row.drop(
# # #         columns=["case_id", "target"]
# # #     )
# # #
# # #     pred = model.predict(input_data)[0]
# # #
# # #     prob = model.predict_proba(input_data)[0][1]
# # #
# # #     st.write("### Prediction Result")
# # #
# # #     if pred == 1:
# # #         st.error(f"⚠️ This claim is likely to be REJECTED")
# # #     else:
# # #         st.success(f"✅ This claim is likely to be APPROVED")
# # #
# # #     st.metric(
# # #         "Rejection Probability",
# # #         f"{prob*100:.2f}%"
# # #     )
# # #
# # #     # =========================================================
# # #     # SHOW CASE EVENTS
# # #     # =========================================================
# # #
# # #     st.subheader("📋 Application Event Trace")
# # #
# # #     case_events = df[df[case_col] == selected_case]
# # #
# # #     st.dataframe(case_events)
# # #
# # #     # =========================================================
# # #     # KPI SECTION
# # #     # =========================================================
# # #
# # #     st.subheader("📌 Process KPIs")
# # #
# # #     col1, col2, col3, col4 = st.columns(4)
# # #
# # #     col1.metric(
# # #         "Total Applications",
# # #         feature_df.shape[0]
# # #     )
# # #
# # #     col2.metric(
# # #         "Rejected Applications",
# # #         feature_df["target"].sum()
# # #     )
# # #
# # #     col3.metric(
# # #         "Average Activities",
# # #         round(feature_df["num_activities"].mean(), 2)
# # #     )
# # #
# # #     col4.metric(
# # #         "Average Duration (Hours)",
# # #         round(feature_df["duration_hours"].mean(), 2)
# # #     )
# # #
# # #     # =========================================================
# # #     # PROCESS VARIANTS
# # #     # =========================================================
# # #
# # #     st.subheader("🔄 Top Process Variants")
# # #
# # #     variant_counts = (
# # #         feature_df["variant"]
# # #         .value_counts()
# # #         .reset_index()
# # #     )
# # #
# # #     variant_counts.columns = ["Variant", "Count"]
# # #
# # #     fig2 = px.bar(
# # #         variant_counts.head(10),
# # #         x="Count",
# # #         y="Variant",
# # #         orientation="h",
# # #         title="Top Variants"
# # #     )
# # #
# # #     st.plotly_chart(fig2, use_container_width=True)
# # #
# # # else:
# # #     st.info("Please upload a CSV file.")
# #
# # import pandas as pd
# #
# # df = pd.read_csv("BPI Challenge 2017_kaggle.csv")
# # df.rename(columns={
# #     'Unnamed: 0': 'Row_ID',
# #     'Action': 'Activity_Action',
# #     'org:resource': 'Resource',
# #     'concept:name': 'Activity_Name',
# #     'EventOrigin': 'Event_Origin',
# #     'EventID': 'Event_ID',
# #     'lifecycle:transition': 'Lifecycle_Transition',
# #     'case:LoanGoal': 'Loan_Goal',
# #     'case:ApplicationType': 'Application_Type',
# #     'case:concept:name': 'Case_ID',
# #     'case:RequestedAmount': 'Requested_Amount',
# #     'FirstWithdrawalAmount': 'First_Withdrawal_Amount',
# #     'NumberOfTerms': 'Number_Of_Terms',
# #     'Accepted': 'Application_Status',
# #     'MonthlyCost': 'Monthly_Cost',
# #     'Selected': 'Offer_Selected',
# #     'CreditScore': 'Credit_Score',
# #     'OfferedAmount': 'Offered_Amount',
# #     'OfferID': 'Offer_ID'
# # }, inplace=True)
# #
# # df1 = df[['Case_ID','Activity_Name','Application_Status']]
# #
# # df1.to_csv("main_data.csv")
#
#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
#     roc_curve
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# # --- STEP 1 & 2: DATA TRANSFORMATION & PREFIX GENERATION ---
# def preprocess_and_prefix(df, max_prefix=10):
#     # Map target
#     df['Application_Status'] = df['Application_Status'].apply(lambda x: 1 if x == 'True' or x is True else 0)
#
#     # Define Target: 1 if SUM(Application_Status) >= 1
#     case_targets = df.groupby('Case_ID')['Application_Status'].max().reset_index()
#     case_targets.columns = ['Case_ID', 'Target']
#
#     # Trace sequences
#     df_sequences = df.groupby('Case_ID')['Activity_Name'].apply(list).reset_index()
#     df_merged = pd.merge(df_sequences, case_targets, on='Case_ID')
#
#     # Generate Prefixes (Length 1 to 10)
#     prefix_data = []
#     for _, row in df_merged.iterrows():
#         case_id, seq, target = row['Case_ID'], row['Activity_Name'], row['Target']
#         for i in range(1, min(len(seq), max_prefix) + 1):
#             prefix_data.append({'Case_ID': case_id, 'Prefix_Length': i, 'Sequence': seq[:i], 'Target': target})
#     return pd.DataFrame(prefix_data)
#
#
# # --- STEP 3: FEATURE ENGINEERING ---
# def extract_features(prefix_df, all_activities):
#     features = []
#     for _, row in prefix_df.iterrows():
#         seq = row['Sequence']
#         feat = {'Case_ID': row['Case_ID'], 'Prefix_Length': row['Prefix_Length'], 'Target': row['Target']}
#         # Position features
#         for i in range(10):
#             feat[f'pos_{i + 1}'] = seq[i] if i < len(seq) else 'NONE'
#         # Frequency features
#         for act in all_activities:
#             feat[f'freq_{act}'] = seq.count(act)
#         # Transition features (Bigrams)
#         if len(seq) > 1:
#             for j in range(len(seq) - 1):
#                 feat[f'trans_{seq[j]}_{seq[j + 1]}'] = 1
#         features.append(feat)
#     return pd.DataFrame(features).fillna(0)
#
#
# # --- EXECUTION ---
# df_raw = pd.read_csv('main_data.csv')
# all_acts = sorted(df_raw['Activity_Name'].unique())
# prefix_df = preprocess_and_prefix(df_raw)
# feat_df = extract_features(prefix_df, all_acts)
#
# # Encode Categorical Positions
# le = LabelEncoder()
# le.fit(all_acts + ['NONE'])
# for i in range(1, 11):
#     feat_df[f'pos_{i}'] = le.transform(feat_df[f'pos_{i}'].astype(str))
#
# # --- STEP 4: TRAINING ---
# X = feat_df.drop(columns=['Case_ID', 'Target'])
# y = feat_df['Target']
# groups = feat_df['Case_ID']
#
# gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
# train_idx, test_idx = next(gss.split(X, y, groups))
# X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
#
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
#
# # --- STEP 5: PREDICTION FUNCTION ---
# def predict_trace(trace, model, le, all_acts, feature_cols):
#     feat = {'Prefix_Length': len(trace)}
#     for i in range(10):
#         feat[f'pos_{i + 1}'] = le.transform([trace[i] if i < len(trace) else 'NONE'])[0]
#     for act in all_acts:
#         feat[f'freq_{act}'] = trace.count(act)
#     # Convert to DF and align
#     df_inf = pd.DataFrame([feat]).reindex(columns=feature_cols, fill_value=0)
#     prob = model.predict_proba(df_inf)[0][1]
#     return prob, "Accepted" if prob >= 0.5 else "Rejected"
#
#
# # Example Usage
# prob, label = predict_trace(["A_Create Application", "A_Submitted", "W_Handle leads"], model, le, all_acts, X.columns)
# print(f"Prob: {prob:.2f}, Result: {label}")
#
#
#
#
#
#
#
#
#
#
#
# ########################################################

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
#
# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Process Prediction Dashboard", layout="wide")
#
# # --- CUSTOM CSS FOR AESTHETICS ---
# st.markdown("""
#     <style>
#     .main { background-color: #f5f7f9; }
#     .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
#     .stAlert { border-radius: 10px; }
#     </style>
#     """, unsafe_allow_html=True)
#
#
# # --- CACHED MODEL TRAINING (Backend) ---
# @st.cache_resource
# def train_model():
#     df = pd.read_csv('main_data.csv')
#     df['Target_Binary'] = df['Application_Status'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
#
#     case_targets = df.groupby('Case_ID')['Target_Binary'].max().reset_index()
#     case_targets.rename(columns={'Target_Binary': 'Target'}, inplace=True)
#     df_sequences = df.groupby('Case_ID')['Activity_Name'].apply(list).reset_index()
#     df_merged = pd.merge(df_sequences, case_targets, on='Case_ID')
#
#     all_acts = sorted(df['Activity_Name'].unique())
#
#     def build_feature_matrix(df_merged, all_acts):
#         rows = []
#         for _, row in df_merged.iterrows():
#             seq = row['Activity_Name']
#             for i in range(1, min(len(seq), 11)):
#                 sub_seq = seq[:i]
#                 feat = {'Case_ID': row['Case_ID'], 'Prefix_Length': i, 'Target': row['Target']}
#                 for p in range(10): feat[f'pos_{p + 1}'] = sub_seq[p] if p < len(sub_seq) else 'NONE'
#                 for act in all_acts: feat[f'freq_{act}'] = sub_seq.count(act)
#                 if len(sub_seq) > 1:
#                     for j in range(len(sub_seq) - 1): feat[f'trans_{sub_seq[j]}_{sub_seq[j + 1]}'] = 1
#                 rows.append(feat)
#         return pd.DataFrame(rows).fillna(0)
#
#     feat_df = build_feature_matrix(df_merged, all_acts)
#     le = LabelEncoder()
#     le.fit(all_acts + ['NONE'])
#     for i in range(1, 11):
#         feat_df[f'pos_{i}'] = le.transform(feat_df[f'pos_{i}'].astype(str))
#
#     X = feat_df.drop(columns=['Case_ID', 'Target'])
#     y = feat_df['Target']
#
#     gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
#     train_idx, test_idx = next(gss.split(X, y, groups=feat_df['Case_ID']))
#     X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
#
#     model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
#     model.fit(X_train, y_train)
#
#     return model, le, all_acts, X_test, y_test, X.columns
#
#
# # Initialize
# model, le, all_acts, X_test, y_test, feature_columns = train_model()
#
# # --- SIDEBAR ---
# st.sidebar.title("🛠️ Model Control Center")
# st.sidebar.markdown("---")
# menu = st.sidebar.radio("Navigate", ["🔮 Prediction Interface", "📊 Model Analytics & Details"])
#
# # --- OPTION 1: PREDICTION INTERFACE ---
# if menu == "🔮 Prediction Interface":
#     st.title("Predictive Process Monitoring")
#     st.subheader("Add event sequence to test the model")
#
#     with st.container():
#         col1, col2 = st.columns([2, 1])
#
#         with col1:
#             user_input = st.multiselect("Select Activity Sequence (Ordered)", all_acts)
#             st.info(f"Prefix Length Detected: {len(user_input)}")
#
#         with col2:
#             st.write("### Result")
#             if st.button("Run Prediction"):
#                 if not user_input:
#                     st.warning("Please select at least one activity.")
#                 else:
#                     # Prepare input
#                     trace = user_input[:10]
#                     feat = {'Prefix_Length': len(trace)}
#                     for i in range(10):
#                         val = trace[i] if i < len(trace) else 'NONE'
#                         feat[f'pos_{i + 1}'] = le.transform([val])[0]
#                     for act in all_acts:
#                         feat[f'freq_{act}'] = trace.count(act)
#
#                     df_inf = pd.DataFrame([feat]).reindex(columns=feature_columns, fill_value=0)
#                     prob = model.predict_proba(df_inf)[0][1]
#
#                     if prob >= 0.5:
#                         st.success(f"**ACCEPTED**")
#                         st.metric("Confidence", f"{prob * 100:.2f}%")
#                     else:
#                         st.error(f"**REJECTED**")
#                         st.metric("Confidence", f"{(1 - prob) * 100:.2f}%")
#
# # --- OPTION 2: MODEL DETAILS ---
# elif menu == "📊 Model Analytics & Details":
#     st.title("Model Performance & Diagnostics")
#
#     tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Confusion Matrix", "Feature Importance"])
#
#     with tab1:
#         y_pred = model.predict(X_test)
#         report = classification_report(y_test, y_pred, output_dict=True)
#         df_report = pd.DataFrame(report).transpose()
#         st.dataframe(df_report.style.background_gradient(cmap='Greens'))
#
#         auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
#         st.metric("Model ROC-AUC Score", f"{auc:.4f}")
#
#     with tab2:
#         fig_cm, ax_cm = plt.subplots()
#         sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', ax=ax_cm)
#         ax_cm.set_title("Confusion Matrix")
#         st.pyplot(fig_cm)
#
#     with tab3:
#         st.write("Top 15 features influencing the prediction:")
#         importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(15)
#         fig_fi, ax_fi = plt.subplots()
#         importances.plot(kind='barh', ax=ax_fi, color='#28a745')
#         plt.gca().invert_yaxis()
#         st.pyplot(fig_fi)
#
# # --- FOOTER ---
# st.sidebar.markdown("---")
# st.sidebar.caption("Senior Data Scientist Framework v1.0")




#
# ###############after  removing add sqquene of activeity feature
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
#
# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Process Analytics AI", layout="wide")
#
# # --- CUSTOM STYLING ---
# st.markdown("""
#     <style>
#     .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
#     .main { background-color: #f8f9fa; }
#     </style>
#     """, unsafe_allow_html=True)
#
#
# # --- BACKEND: MODEL TRAINING (Cachced for speed) ---
# @st.cache_resource
# def load_and_train():
#     # Dataset Loading
#     df = pd.read_csv('main_data.csv')
#     df['Target_Binary'] = df['Application_Status'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
#
#     # Target Generation
#     case_targets = df.groupby('Case_ID')['Target_Binary'].max().reset_index()
#     case_targets.rename(columns={'Target_Binary': 'Target'}, inplace=True)
#     df_sequences = df.groupby('Case_ID')['Activity_Name'].apply(list).reset_index()
#     df_merged = pd.merge(df_sequences, case_targets, on='Case_ID')
#
#     all_acts = sorted(df['Activity_Name'].unique())
#
#     # Feature Engineering Logic
#     def build_features(df_merged, all_acts):
#         rows = []
#         for _, row in df_merged.iterrows():
#             seq = row['Activity_Name']
#             for i in range(1, min(len(seq), 11)):
#                 sub_seq = seq[:i]
#                 feat = {'Case_ID': row['Case_ID'], 'Prefix_Length': i, 'Target': row['Target']}
#                 for p in range(10): feat[f'pos_{p + 1}'] = sub_seq[p] if p < len(sub_seq) else 'NONE'
#                 for act in all_acts: feat[f'freq_{act}'] = sub_seq.count(act)
#                 rows.append(feat)
#         return pd.DataFrame(rows).fillna(0)
#
#     feat_df = build_features(df_merged, all_acts)
#     le = LabelEncoder()
#     le.fit(all_acts + ['NONE'])
#     for i in range(1, 11): feat_df[f'pos_{i}'] = le.transform(feat_df[f'pos_{i}'].astype(str))
#
#     X = feat_df.drop(columns=['Case_ID', 'Target'])
#     y = feat_df['Target']
#
#     # Model Training
#     model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
#     model.fit(X, y)
#
#     return model, le, all_acts, X.columns
#
#
# # Init backend
# model, le, all_acts, feature_cols = load_and_train()
#
# # --- SIDEBAR NAVIGATION ---
# st.sidebar.header("Navigation")
# page = st.sidebar.radio("Go to", ["🔍 Prediction Dashboard", "📈 Model Insights"])
#
# # --- PAGE 1: PREDICTION DASHBOARD ---
# if page == "🔍 Prediction Dashboard":
#     st.title("Process Predictive Monitoring")
#     st.info("Naye case ka outcome predict karne ke liye uski activity history CSV format mein upload karein.")
#
#     # File Upload Section
#     col1, col2 = st.columns([1, 1])
#
#     with col1:
#         st.subheader("Upload Case Data")
#         uploaded_file = st.file_uploader("Choose CSV (Should contain 'Activity_Name' column)", type="csv")
#
#         if uploaded_file:
#             input_df = pd.read_csv(uploaded_file)
#             st.write("**Sequence Detected:**")
#             st.dataframe(input_df[['Activity_Name']].head(10), use_container_width=True)
#
#     with col2:
#         st.subheader("Final Prediction")
#         if uploaded_file:
#             # Inference Logic
#             trace = input_df['Activity_Name'].tolist()[:10]
#
#             # Prepare Input Feature Vector
#             feat = {'Prefix_Length': len(trace)}
#             for i in range(10):
#                 val = trace[i] if i < len(trace) else 'NONE'
#                 try:
#                     feat[f'pos_{i + 1}'] = le.transform([val])[0]
#                 except:
#                     feat[f'pos_{i + 1}'] = le.transform(['NONE'])[0]
#
#             for act in all_acts:
#                 feat[f'freq_{act}'] = trace.count(act)
#
#             # Prediction
#             df_inf = pd.DataFrame([feat]).reindex(columns=feature_cols, fill_value=0)
#             prob = model.predict_proba(df_inf)[0][1]
#
#             # Display Results Aesthetically
#             st.markdown("---")
#             if prob >= 0.5:
#                 st.success("### STATUS: LIKELY ACCEPTED")
#                 st.metric(label="Acceptance Confidence", value=f"{prob * 100:.1f}%")
#             else:
#                 st.error("### STATUS: LIKELY REJECTED")
#                 st.metric(label="Rejection Confidence", value=f"{(1 - prob) * 100:.1f}%")
#
#             st.progress(prob)
#         else:
#             st.write("Waiting for data upload...")
#
# # --- PAGE 2: MODEL INSIGHTS ---
# elif page == "📈 Model Insights":
#     st.title("Model Diagnostics & Explainability")
#
#     col_l, col_r = st.columns(2)
#
#     with col_l:
#         st.subheader("Activity Influence")
#         # Feature Importance Plot
#         importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.barplot(x=importances.values, y=importances.index, palette="Blues_d", ax=ax)
#         plt.title("Top 12 Predictive Features")
#         st.pyplot(fig)
#
#     with col_r:
#         st.subheader("Model Parameters")
#         st.json({
#             "Algorithm": "Random Forest (Traditional ML)",
#             "Sequence Window": "First 10 Activities",
#             "Class Balancing": "SMOTE + Balanced Weights",
#             "N_Estimators": 150,
#             "Target Leakage Protection": "Case-level Group Splitting"
#         })
#
#
# # Footer
# st.sidebar.markdown("---")
# st.sidebar.caption("Powered by Senior Process Mining Engine")



#####################$after including the deteils in insifht part
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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