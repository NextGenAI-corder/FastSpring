import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import shap
import pickle
import sys

# CSV file
csv_filename = "./data/credit_data_learn.csv"
df = pd.read_csv(csv_filename, encoding="utf-8-sig")
df.columns = df.columns.str.replace(r"[\s\t\r\n\uFEFF]", "", regex=True)
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

# Required columns list
required_cols = [
    "Name",
    "Sex",
    "Age",
    "Marital",
    "Income",
    "CreditAppAmount",
    "OtherDebts",
    "DelinquencyInfo",
    "DebtRestruct",
    "EmploymentYears",
]

# Check for missing required columns
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Error Missing required columns: {missing_cols}")

# Check for null values in required columns
null_info = {
    c: df[df[c].isnull()].index.tolist() for c in required_cols if df[c].isnull().any()
}
if null_info:
    for col, idxs in null_info.items():
        print(f"Error Missing values: {col} → Row indices: {idxs}")
    sys.exit()

# Define expected values for categorical variables
valid_sex = {"Man", "Woman"}
valid_marital = {"Single", "Married"}

# Check for invalid categorical values (including blanks and nulls)
invalid_sex = df[~df["Sex"].isin(valid_sex)]["Sex"]
invalid_marital = df[~df["Marital"].isin(valid_marital)]["Marital"]

if not invalid_sex.empty:
    print(f"Error Invalid values in Sex column: {invalid_sex.unique().tolist()}")
    sys.exit()

if not invalid_marital.empty:
    print(
        f"Error Invalid values in Marital column: {invalid_marital.unique().tolist()}"
    )
    sys.exit()

# Input value validation
bad_cols = []
if ((df["Age"] <= 18) | (df["Age"] >= 80)).any():
    bad_cols.append("Age(18<Age<80)")
if (df["Income"] <= 0).any():
    bad_cols.append("Income(>0)")
if (df["CreditAppAmount"] <= 0).any():
    bad_cols.append("CreditAppAmount(>0)")
if (df["OtherDebts"] < 0).any():
    bad_cols.append("OtherDebts(>=0)")
if (df["EmploymentYears"] < 0).any():
    bad_cols.append("EmploymentYears(>=0)")
if (df["EmploymentYears"] > (df["Age"] - 15)).any():
    bad_cols.append("EmploymentYears<=Age-15")
if bad_cols:
    raise ValueError(f"Error Invalid values: {bad_cols}")

# Calculate BorrowingRatio (Debt to Income Ratio)
if "BorrowingRatio" not in df.columns:
    df["BorrowingRatio"] = np.where(
        (df["Income"] > 0),
        (df["OtherDebts"]) / df["Income"],
        0.0,
    )

if "BorrowingRatio" not in df.columns:
    df["BorrowingRatio"] = np.where(
        (df["Income"] > 0),
        (df["OtherDebts"]) / df["Income"],
        0.0,
    )

if not np.issubdtype(df["BorrowingRatio"].dtype, np.number):
    print("Error: BorrowingRatio contains non-numeric values")
    sys.exit()

# Encode categorical variables
cat_cols = ["Sex", "Marital", "Occupation", "Industry"]
cat_maps = {}
for col in cat_cols:
    df[col] = df[col].astype("category")
    cat_maps[col] = dict(
        zip(df[col].cat.categories, range(len(df[col].cat.categories)))
    )
    df[col] = df[col].cat.codes

with open("cat_maps.pkl", "wb") as f:
    pickle.dump(cat_maps, f)

y = df["DelinquencyInfo"].fillna(0)

# Backup Name column before drop
name_col = df["Name"].copy()

# X = df.drop(["DelinquencyInfo", "Name","BorrowingRatio"], axis=1)
X = df.drop(["DelinquencyInfo", "Name"], axis=1)
# X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
X = X.astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Japanese to English feature name mapping
"""feature_jp_map = {
    "Name": "名前",
    "Sex": "性別",
    "Marital": "婚姻状況",
    "Age": "年齢",
    "Income": "年収",
    "CreditAppAmount": "申込額",
    "OtherDebts": "他債務",
    "DebtRestruct": "債務整理",
    "BorrowingRatio": "借入比率",
    "JobType": "職種区分",
    "Occupation": "職業",
    "Industry": "業種",
    "Education": "学歴",
    "Dependents": "扶養人数",
    "OwnHouse": "持ち家",
    "Foreigner": "外国人",
    "Phone": "電話有無",
    "EmploymentYears": "勤続年数",
    "Guarantor": "保証人有無",
    "Collateral": "担保有無",
}"""

# Weight adjustment
# Prioritize Recall ⇒ Increase (e.g., 25, 30...)
# Prioritize Precision ⇒ Decrease (e.g., 15, 10...)
class_weight = {0: 1.0, 1: 6}

# Build the model
model = lgb.LGBMClassifier(
    objective="binary",  # Binary classification (e.g., delinquent or not)
    random_state=42,  # Seed for reproducibility
    verbosity=-1,  # Suppress training logs
    # Model complexity and stability
    n_estimators=700,
    learning_rate=0.0055,
    # Class weighting for imbalance
    class_weight=class_weight,
    feature_fraction=0.8,
    reg_alpha=1,
    reg_lambda=1,
    max_depth=4,
    min_child_samples=8,
    subsample=1,
    colsample_bytree=0.7,
)

model.fit(X_train, y_train)

# Inference and evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Get SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_vals_individual = shap_values[1] if isinstance(shap_values, list) else shap_values

# SHAP values and scores per individual
shap_df = pd.DataFrame(X_test)
# shap_df["Name"] = name_col.iloc[X_test.index].values
shap_df["Name"] = name_col.iloc[X_test.index].values + 1

shap_df["Score"] = y_pred_proba
shap_df_sorted = shap_df.sort_values("Name").reset_index(drop=True)

# Output SHAP per Name order
for i, row in enumerate(shap_df_sorted.itertuples(), start=1):
    shap_series = pd.Series(shap_vals_individual[row.Index], index=X_test.columns)
    shap_top3 = shap_series.abs().sort_values(ascending=False).head(3)

    print(f"\nIndividual Name: {int(row.Name)} Score: {row.Score:.4f}")
    print("Top 3 SHAP Features")
    for feat in shap_top3.index:
        val = shap_series[feat]
        sign = "Positive" if val > 0 else "Negative"
        # jp = feature_jp_map.get(feat, feat)
        # print(f"{jp}: SHAP = {val:.4f} ({sign})")
        print(f"{feat}: SHAP = {val:.4f} ({sign})")

# Overall evaluation
print("\nOverall Model Evaluation")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# Export credit evaluation results
results = X_test.copy()
results.insert(0, "Name", name_col.iloc[X_test.index].values)
results = results.sort_values("Name")

results["PredictedProbability"] = y_pred_proba
results["Actual"] = y_test.values

# Restore original order before exporting
results = results.sort_index()
results.to_csv("individual_scores.csv", index=False, encoding="utf-8-sig")

# Output formatted data for input.csv (pre-model training stage)
df_out = X.copy()
df_out.insert(0, "Name", df["Name"])
df_out.insert(7, "DelinquencyInfo", y)
df_out.to_csv("./data/input.csv", index=False, encoding="utf-8-sig")
print("Formatted credit evaluation data has been output to input.csv.")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel and scores have been saved to individual_scores.csv.")
