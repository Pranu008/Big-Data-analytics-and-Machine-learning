
# Linear Regression Model for Weekly Sales Prediction
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

# STEP 1: Load Provided Dataset
data = pd.read_csv("weekly_sales_dataset.csv")
print("\nDataset Preview:")
print(data.head())

# Target variable
target = "Weekly_Sales"
# STEP 2: Out-of-Time Split
# Since no date column exists, we assume data is ordered by time
split_index = int(len(data) * 0.8)

train = data.iloc[:split_index]
test = data.iloc[split_index:]

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nOut-of-Time Validation Results")
print("MAE:", round(mae, 2))
print("R²:", round(r2, 2))

# STEP 3: Deprioritized Hyperparameter Tuning
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=LinearRegression(),
    param_grid={},  # No major hyperparameters for Linear Regression
    cv=cv,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("\nCross-Validation Score (Not Prioritized):", round(grid.best_score_, 2))
# STEP 4: Business-Friendly Coefficients
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Impact_on_Weekly_Sales": model.coef_
})

print("\nModel Coefficients (Business Interpretation):")
print(coefficients)
