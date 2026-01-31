
#1. Baseline Linear Regression Model with Diagnostics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("weekly_sales_dataset.csv")
print("\nDataset Preview:")
print(data.head())

X = data.drop(columns=["Weekly_Sales"])
y = data["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nBaseline Linear Regression Performance")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Impact_on_Weekly_Sales": model.coef_
})

print("\nModel Coefficients (Business Interpretation):")
print(coef_df)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Predicted vs Actual Weekly Sales")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residual Errors")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Sales")
plt.show()

print("\nResidual Mean:", round(residuals.mean(), 2))
print("Residual Std Dev:", round(residuals.std(), 2))

# 2. Counterfactual Experiment: Increase Advertising Spend by 20%
X = data.drop(columns=["Weekly_Sales"])
y = data["Weekly_Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train baseline model
model = LinearRegression()
model.fit(X_train, y_train)
# Baseline predictions
baseline_preds = model.predict(X_test)

X_test_cf = X_test.copy()
X_test_cf["Advertising_Spend"] = X_test_cf["Advertising_Spend"] * 1.20

counterfactual_preds = model.predict(X_test_cf)

comparison_df = pd.DataFrame({
    "Original_Ad_Spend": X_test["Advertising_Spend"],
    "Increased_Ad_Spend": X_test_cf["Advertising_Spend"],
    "Baseline_Predicted_Sales": baseline_preds,
    "Counterfactual_Predicted_Sales": counterfactual_preds,
})

comparison_df["Predicted_Sales_Change"] = (
    comparison_df["Counterfactual_Predicted_Sales"] -
    comparison_df["Baseline_Predicted_Sales"]
)

print("\nCounterfactual Experiment Results (First 5 Rows):")
print(comparison_df.head())

avg_sales_increase = comparison_df["Predicted_Sales_Change"].mean()
avg_ad_increase = (X_test_cf["Advertising_Spend"] - X_test["Advertising_Spend"]).mean()

roi_ratio = avg_sales_increase / avg_ad_increase

print("\nAverage Increase in Advertising Spend:", round(avg_ad_increase, 2))
print("Average Predicted Increase in Sales:", round(avg_sales_increase, 2))
print("Predicted Sales Increase per 1 Unit of Ad Spend:", round(roi_ratio, 2))

print("\nInterpretation:")
if roi_ratio > 1:
    print("The model suggests strong return on advertising — economically plausible if margins support it.")
elif 0 < roi_ratio <= 1:
    print("The model suggests modest returns — plausible but depends on profit margins.")
else:
    print("The model suggests negative or unrealistic impact — may indicate model or data issues.")
# Baseline Linear Regression Model with Diagnostics


#3. Model Misspecification Experiment
model_correct = LinearRegression()
model_correct.fit(X_train, y_train)

preds_correct = model_correct.predict(X_test)
mae_correct = mean_absolute_error(y_test, preds_correct)
r2_correct = r2_score(y_test, preds_correct)

coef_correct = pd.Series(model_correct.coef_, index=X.columns)

X_train_miss = X_train.drop(columns=["Advertising_Spend"])
X_test_miss = X_test.drop(columns=["Advertising_Spend"])

model_miss = LinearRegression()
model_miss.fit(X_train_miss, y_train)

preds_miss = model_miss.predict(X_test_miss)
mae_miss = mean_absolute_error(y_test, preds_miss)
r2_miss = r2_score(y_test, preds_miss)

coef_miss = pd.Series(model_miss.coef_, index=X_train_miss.columns)

print("\n=== Model Performance Comparison ===")
print("Correct Model MAE:", round(mae_correct, 2))
print("Misspecified Model MAE:", round(mae_miss, 2))
print("Correct Model R²:", round(r2_correct, 2))
print("Misspecified Model R²:", round(r2_miss, 2))

print("\n=== Coefficient Comparison ===")
print("Correct Model Coefficients:")
print(coef_correct)

print("\nMisspecified Model Coefficients (Advertising removed):")
print(coef_miss)

# Prediction difference
prediction_diff = np.mean(np.abs(preds_correct - preds_miss))
print("\nAverage Difference in Predictions:", round(prediction_diff, 2))

print("\nBusiness Risk Interpretation:")
print("Removing Advertising_Spend causes other variables to absorb its effect.")
print("This distorts the price and competitor price impact, leading to biased decisions.")
print("Business may underinvest in advertising or misinterpret price sensitivity.")
