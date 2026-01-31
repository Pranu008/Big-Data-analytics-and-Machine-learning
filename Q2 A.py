import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("weekly_sales_dataset.csv")
# Features (inputs) and target (output)
X = data[["Advertising_Spend", "Price", "Competitor_Price"]]
y = data["Weekly_Sales"]

# Scale features because KNN depends on distance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model R² Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

"""
IMPLICIT ASSUMPTIONS:

1. Similar inputs lead to similar outputs
   - If two weeks have similar prices and ad spend, sales will be similar.

2. Distance is meaningful
   - KNN relies on distance, so features must be scaled.

3. All features influence the result equally unless weighted
   - The algorithm treats each feature as equally important.

4. Data should not have too many dimensions
   - Too many features reduce distance effectiveness (curse of dimensionality).
"""
"""
BUSINESS SCENARIO: LOAN APPROVAL

A bank compares two models:

Model A (More Accurate, Complex)
- Uses advanced algorithms
- Slightly better at predicting loan defaults
- Hard to explain decisions

Model B (Weaker, Simple)
- Uses logistic regression or a decision tree
- Slightly less accurate
- Can explain decisions using income, credit score, and debt

WHY MODEL B MAY BE CHOSEN:

- Banking laws require clear explanations
- Regulators may audit the system
- Complex models may hide bias
- Legal and reputation risks outweigh small accuracy gains
"""
"""
Loan Approval Prediction Example
--------------------------------
This script compares:
1. A complex, more accurate model (Random Forest)
2. A simpler, more explainable model (Logistic Regression)

Goal: Show why a bank might choose the weaker but more transparent model.
"""

# Example dataset
data = pd.DataFrame({
    "Income": [25000, 50000, 75000, 100000, 30000, 80000, 120000, 45000, 60000, 90000],
    "Credit_Score": [600, 700, 750, 800, 580, 720, 810, 690, 710, 770],
    "Loan_Amount": [10000, 20000, 25000, 30000, 12000, 22000, 35000, 18000, 24000, 28000],
    "Approved": [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]  # 0 = Rejected, 1 = Approved
})

X = data[["Income", "Credit_Score", "Loan_Amount"]]
y = data["Approved"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_preds))

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_preds))

feature_names = ["Income", "Credit_Score", "Loan_Amount"]
coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Impact_on_Approval": lr_model.coef_[0]
})
print("\nLogistic Regression Feature Impact:")
print(coefficients)
