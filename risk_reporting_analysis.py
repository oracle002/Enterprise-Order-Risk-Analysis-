import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sns.set_theme(style="whitegrid")
PLOT_DIR = "plots"
pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# 1. DATA INGESTION & VALIDATION (data quality / integrity checks)
# ---------------------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/synthetic_ecommerce_order_risk_dataset.csv")
df["order_date"] = pd.to_datetime(df["order_date"])

validation_log = {
    "total_records": len(df),
    "duplicate_order_ids": df["order_id"].duplicated().sum(),
    "missing_values": int(df.isna().sum().sum()),
    "negative_order_values": (df["order_value_eur"] < 0).sum(),
    "review_score_out_of_range": ((df["review_score"] < 0) | (df["review_score"] > 5)).sum(),
    "date_range": f"{df['order_date'].min().date()} to {df['order_date'].max().date()}",
}
print("=== DATA VALIDATION LOG ===")
for k, v in validation_log.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------------------------
# 2. RISK INDICATOR AGGREGATION (BCBS-239-style aggregated risk reporting)
# ---------------------------------------------------------------------------
df["month"] = df["order_date"].dt.to_period("M").astype(str)
df["is_high_risk"] = (df["risk_label"] != "Normal").astype(int)

overall_rates = {
    "fraud_rate_%": round(df["is_fraud"].mean() * 100, 2),
    "return_rate_%": round(df["is_returned"].mean() * 100, 2),
    "combined_high_risk_rate_%": round(df["is_high_risk"].mean() * 100, 2),
}
print("\n=== ENTERPRISE-LEVEL RISK APPETITE METRICS ===")
print(overall_rates)

# Risk concentration scorecard by key dimensions (mirrors a risk reporting pack)
dimensions = ["country", "payment_method", "product_category", "device_type", "traffic_source"]
scorecards = {}
for dim in dimensions:
    scorecard = df.groupby(dim).agg(
        orders=("order_id", "count"),
        fraud_rate_pct=("is_fraud", lambda x: round(x.mean() * 100, 2)),
        return_rate_pct=("is_returned", lambda x: round(x.mean() * 100, 2)),
        avg_order_value=("order_value_eur", "mean"),
        avg_review_score=("review_score", "mean"),
    ).sort_values("fraud_rate_pct", ascending=False)
    scorecards[dim] = scorecard
    print(f"\n=== RISK SCORECARD BY {dim.upper()} ===")
    print(scorecard.round(2))

# Flag dimension segments exceeding a defined risk appetite threshold (effective challenge)
THRESHOLD_MULTIPLIER = 2.0  # flag if segment fraud rate > 2x overall average
overall_fraud_rate = df["is_fraud"].mean() * 100
breaches = []
for dim, sc in scorecards.items():
    flagged = sc[sc["fraud_rate_pct"] > overall_fraud_rate * THRESHOLD_MULTIPLIER]
    for idx, row in flagged.iterrows():
        breaches.append({"dimension": dim, "segment": idx, "fraud_rate_pct": row["fraud_rate_pct"],
                          "orders": row["orders"]})
breach_log = pd.DataFrame(breaches).sort_values("fraud_rate_pct", ascending=False)
print("\n=== RISK APPETITE BREACH LOG (segments > 2x average fraud rate) ===")
print(breach_log)
breach_log.to_csv("risk_appetite_breach_log.csv", index=False)

monthly_trend = df.groupby("month").agg(
    orders=("order_id", "count"),
    fraud_rate_pct=("is_fraud", lambda x: round(x.mean() * 100, 2)),
    return_rate_pct=("is_returned", lambda x: round(x.mean() * 100, 2)),
).reset_index()
monthly_trend.to_csv("monthly_risk_trend.csv", index=False)

# Save consolidated risk reporting pack
with pd.ExcelWriter("risk_reporting_pack.xlsx") as writer:
    pd.DataFrame([overall_rates]).to_excel(writer, sheet_name="Enterprise Summary", index=False)
    monthly_trend.to_excel(writer, sheet_name="Monthly Trend", index=False)
    breach_log.to_excel(writer, sheet_name="Appetite Breaches", index=False)
    for dim, sc in scorecards.items():
        sc.to_excel(writer, sheet_name=f"By {dim}"[:31])

# ---------------------------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
df["risk_label"].value_counts().plot(kind="bar", color=["#2E7D32", "#F9A825", "#C62828"], ax=ax)
ax.set_title("Order Risk Label Distribution")
ax.set_xlabel("Risk Label")
ax.set_ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/risk_label_distribution.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
monthly_trend.plot(x="month", y=["fraud_rate_pct", "return_rate_pct"], ax=ax, marker="o")
ax.set_title("Monthly Fraud & Return Rate Trend")
ax.set_ylabel("Rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/monthly_risk_trend.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
sc = scorecards["payment_method"].sort_values("fraud_rate_pct")
ax.barh(sc.index, sc["fraud_rate_pct"], color="#C62828")
ax.set_title("Fraud Rate (%) by Payment Method")
ax.set_xlabel("Fraud Rate (%)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fraud_by_payment_method.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
sc2 = scorecards["country"].sort_values("return_rate_pct")
ax.barh(sc2.index, sc2["return_rate_pct"], color="#F9A825")
ax.set_title("Return Rate (%) by Country")
ax.set_xlabel("Return Rate (%)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/return_rate_by_country.png", dpi=150)
plt.close()

corr_cols = ["customer_age_days", "previous_orders", "avg_order_value_eur", "order_value_eur",
             "discount_rate", "shipping_distance_km", "delivery_days_estimated",
             "late_delivery_risk", "address_mismatch", "high_risk_ip",
             "customer_support_contacts", "review_score", "is_returned", "is_fraud"]
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df[corr_cols].corr(), cmap="RdBu_r", center=0, annot=False, ax=ax)
ax.set_title("Correlation Matrix of Risk-Relevant Features")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. PREDICTIVE RISK CLASSIFICATION MODEL
# ---------------------------------------------------------------------------
model_df = df.copy()
cat_cols = ["country", "device_type", "traffic_source", "payment_method", "product_category"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    model_df[c + "_enc"] = le.fit_transform(model_df[c])
    encoders[c] = le

feature_cols = [c + "_enc" for c in cat_cols] + [
    "customer_age_days", "previous_orders", "avg_order_value_eur", "order_value_eur",
    "quantity", "discount_rate", "shipping_distance_km", "delivery_days_estimated",
    "late_delivery_risk", "address_mismatch", "high_risk_ip",
    "customer_support_contacts", "review_score",
]

X = model_df[feature_cols]
y = model_df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== RISK CLASSIFICATION MODEL PERFORMANCE (RandomForest) ===")
report = classification_report(y_test, y_pred)
print(report)
with open("model_classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
ax.set_title("Risk Label Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# Fraud-only binary AUC (secondary target)
y_fraud = model_df["is_fraud"]
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X, y_fraud, test_size=0.25, random_state=42, stratify=y_fraud
)
clf_fraud = RandomForestClassifier(
    n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
)
clf_fraud.fit(Xf_train, yf_train)
fraud_proba = clf_fraud.predict_proba(Xf_test)[:, 1]
auc = roc_auc_score(yf_test, fraud_proba)
print(f"\nFraud detection model ROC-AUC: {auc:.3f}")

# Feature importance (drives the reporting narrative on top risk drivers)
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== TOP RISK DRIVERS (Feature Importance) ===")
print(importances.head(10))
importances.to_csv("feature_importance.csv")

fig, ax = plt.subplots(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh", color="#1565C0", ax=ax)
ax.set_title("Top 10 Risk Drivers (Feature Importance)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=150)
plt.close()

print("\nDone. Reporting pack, breach log, and plots saved.")
