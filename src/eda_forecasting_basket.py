"""
Extension: Exploratory Data Analysis, Risk Forecasting, and Market Basket
(Association Rule) Analysis
--------------------------------------------------------------------------
Builds on risk_reporting_analysis.py. Adds three components commonly asked
for in analyst/risk-reporting portfolios:

  A. Deeper EDA — distributions, outliers, bivariate risk relationships
  B. Forecasting — monthly order volume & risk-rate trend forecast
  C. Market Basket / Association Rule Mining — adapted for this dataset,
     since each row is a single-product order (no multi-item basket or
     customer ID exists to group repeat purchases). Instead of classic
     "items bought together", this treats each order's categorical
     attributes (country, payment method, category, channel, device,
     risk flags) as a transaction "basket" and mines which COMBINATIONS
     of attributes co-occur most strongly with fraud/return outcomes —
     the standard adaptation of Apriori/association-rule mining used in
     fraud-pattern detection when no item-level basket exists.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mlxtend.frequent_patterns import apriori, association_rules

sns.set_theme(style="whitegrid")
PLOT_DIR = "../reports/plots"

df = pd.read_csv("../data/synthetic_ecommerce_order_risk_dataset.csv")
df["order_date"] = pd.to_datetime(df["order_date"])
df["month"] = df["order_date"].dt.to_period("M")

# ===========================================================================
# A. EXPLORATORY DATA ANALYSIS
# ===========================================================================
print("=== A. EXPLORATORY DATA ANALYSIS ===")

numeric_cols = ["order_value_eur", "avg_order_value_eur", "discount_rate",
                 "shipping_distance_km", "delivery_days_estimated",
                 "customer_age_days", "previous_orders", "review_score",
                 "customer_support_contacts"]

summary = df[numeric_cols].describe().T
summary.to_csv("../reports/eda_numeric_summary.csv")
print(summary.round(2))

# --- Distribution grid ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
dist_cols = ["order_value_eur", "discount_rate", "delivery_days_estimated",
             "customer_age_days", "review_score", "shipping_distance_km"]
for ax, col in zip(axes.flat, dist_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#1565C0")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_distributions.png", dpi=150)
plt.close()

# --- Outlier detection (IQR method) ---
outlier_report = {}
for col in ["order_value_eur", "shipping_distance_km", "delivery_days_estimated"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = {"lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
                            "n_outliers": int(n_out), "pct_outliers": round(n_out / len(df) * 100, 2)}
outlier_df = pd.DataFrame(outlier_report).T
outlier_df.to_csv("../reports/eda_outlier_report.csv")
print("\n--- Outlier report (IQR method) ---")
print(outlier_df)

# --- Bivariate: order value & review score by risk label ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="risk_label", y="order_value_eur", ax=axes[0], palette="Set2")
axes[0].set_title("Order Value by Risk Label")
sns.boxplot(data=df, x="risk_label", y="review_score", ax=axes[1], palette="Set2")
axes[1].set_title("Review Score by Risk Label")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_boxplots_by_risk.png", dpi=150)
plt.close()

# --- Categorical order volume breakdown ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.flat, ["country", "device_type", "traffic_source", "payment_method"]):
    df[col].value_counts().plot(kind="bar", ax=ax, color="#455A64")
    ax.set_title(f"Orders by {col}")
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_categorical_breakdown.png", dpi=150)
plt.close()

# --- Risk rate by device type (bivariate categorical vs risk) ---
device_risk = df.groupby("device_type")[["is_fraud", "is_returned"]].mean() * 100
print("\n--- Risk rate (%) by device type ---")
print(device_risk.round(2))

# ===========================================================================
# B. FORECASTING — monthly order volume & risk rate trend
# ===========================================================================
print("\n=== B. FORECASTING ===")

monthly = df.groupby("month").agg(
    orders=("order_id", "count"),
    fraud_rate=("is_fraud", "mean"),
    return_rate=("is_returned", "mean"),
).reset_index()
monthly["month"] = monthly["month"].astype(str)
monthly.set_index(pd.PeriodIndex(monthly["month"], freq="M").to_timestamp(), inplace=True)

# Seasonal decomposition of order volume
decomp = seasonal_decompose(monthly["orders"], model="additive", period=12)
fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/forecast_seasonal_decomposition.png", dpi=150)
plt.close()

# Holt-Winters forecast: order volume (trend + yearly seasonality), 3-month horizon
hw_orders = ExponentialSmoothing(
    monthly["orders"], trend="add", seasonal="add", seasonal_periods=12
).fit()
forecast_horizon = 3
orders_forecast = hw_orders.forecast(forecast_horizon)

# Simple exponential smoothing for fraud & return rate (noisier, less seasonal signal)
hw_fraud = ExponentialSmoothing(monthly["fraud_rate"], trend="add").fit()
hw_return = ExponentialSmoothing(monthly["return_rate"], trend="add").fit()
fraud_forecast = hw_fraud.forecast(forecast_horizon)
return_forecast = hw_return.forecast(forecast_horizon)

forecast_table = pd.DataFrame({
    "forecast_month": orders_forecast.index.strftime("%Y-%m"),
    "forecasted_orders": orders_forecast.values.round(0),
    "forecasted_fraud_rate_%": (fraud_forecast.values * 100).round(2),
    "forecasted_return_rate_%": (return_forecast.values * 100).round(2),
})
forecast_table.to_csv("../reports/forecast_next_3_months.csv", index=False)
print(forecast_table)

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
axes[0].plot(monthly.index, monthly["orders"], label="Actual", marker="o")
axes[0].plot(orders_forecast.index, orders_forecast.values, label="Forecast", marker="o", linestyle="--", color="red")
axes[0].set_title("Monthly Order Volume — Actual vs. 3-Month Forecast")
axes[0].legend()

axes[1].plot(monthly.index, monthly["fraud_rate"] * 100, label="Fraud Rate Actual", marker="o", color="#C62828")
axes[1].plot(fraud_forecast.index, fraud_forecast.values * 100, label="Fraud Rate Forecast", marker="o", linestyle="--", color="#C62828", alpha=0.5)
axes[1].plot(monthly.index, monthly["return_rate"] * 100, label="Return Rate Actual", marker="o", color="#F9A825")
axes[1].plot(return_forecast.index, return_forecast.values * 100, label="Return Rate Forecast", marker="o", linestyle="--", color="#F9A825", alpha=0.5)
axes[1].set_title("Monthly Fraud & Return Rate — Actual vs. 3-Month Forecast")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/forecast_orders_and_risk_rates.png", dpi=150)
plt.close()

# ===========================================================================
# C. MARKET BASKET / ASSOCIATION RULE MINING (attribute-based, fraud-pattern focus)
# ===========================================================================
print("\n=== C. MARKET BASKET / ASSOCIATION RULE ANALYSIS ===")
print("Note: dataset has one product per order and no customer ID linking repeat")
print("purchases, so classic item-co-purchase basket analysis isn't available.")
print("Instead, order attributes are treated as basket 'items' to mine which")
print("combinations most strongly associate with fraud/return outcomes.\n")

basket_df = pd.DataFrame(index=df.index)
basket_df["country=" + df["country"]] = True
basket_df["payment=" + df["payment_method"]] = True
basket_df["category=" + df["product_category"]] = True
basket_df["channel=" + df["traffic_source"]] = True
basket_df["device=" + df["device_type"]] = True
basket_df["high_risk_ip"] = df["high_risk_ip"] == 1
basket_df["address_mismatch"] = df["address_mismatch"] == 1
basket_df["late_delivery_risk"] = df["late_delivery_risk"] == 1
basket_df["is_fraud"] = df["is_fraud"] == 1
basket_df["is_returned"] = df["is_returned"] == 1

# one-hot encode the categorical "item" columns properly (each row True only in its own category)
onehot = pd.get_dummies(df[["country", "payment_method", "product_category",
                             "traffic_source", "device_type"]].astype(str))
onehot = onehot.astype(bool)
onehot["high_risk_ip"] = df["high_risk_ip"] == 1
onehot["address_mismatch"] = df["address_mismatch"] == 1
onehot["late_delivery_risk"] = df["late_delivery_risk"] == 1
onehot["is_fraud"] = df["is_fraud"] == 1
onehot["is_returned"] = df["is_returned"] == 1

frequent_itemsets = apriori(onehot, min_support=0.004, use_colnames=True, max_len=3)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Focus on rules whose consequent is the fraud or return outcome
fraud_rules = rules[rules["consequents"].apply(lambda x: "is_fraud" in x)].copy()
fraud_rules = fraud_rules.sort_values("lift", ascending=False)
fraud_rules["antecedents"] = fraud_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
fraud_rules["consequents"] = fraud_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
fraud_rules = fraud_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
fraud_rules.to_csv("../reports/market_basket_fraud_rules.csv", index=False)
print("--- Top association rules leading to FRAUD (by lift) ---")
print(fraud_rules.head(10).round(3))

return_rules = rules[rules["consequents"].apply(lambda x: "is_returned" in x)].copy()
return_rules = return_rules.sort_values("lift", ascending=False)
return_rules["antecedents"] = return_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
return_rules["consequents"] = return_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
return_rules = return_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
return_rules.to_csv("../reports/market_basket_return_rules.csv", index=False)
print("\n--- Top association rules leading to RETURNS (by lift) ---")
print(return_rules.head(10).round(3))

# Plot top fraud rules by lift
top_fraud = fraud_rules.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_fraud["antecedents"], top_fraud["lift"], color="#C62828")
ax.set_xlabel("Lift")
ax.set_title("Top Attribute Combinations Associated with Fraud (Lift)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/market_basket_fraud_rules.png", dpi=150)
plt.close()

print("\nDone. EDA, forecast, and market-basket outputs saved.")
