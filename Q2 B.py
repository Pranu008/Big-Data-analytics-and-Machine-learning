from statistics import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("weekly_sales_dataset.csv")

X = data[["Advertising_Spend", "Price", "Competitor_Price"]]
y = data["Weekly_Sales"]

# Scale features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
# Define model
knn = KNeighborsRegressor()

# Define hyperparameter grid (tuning number of neighbors)
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11]
}

# Grid search with cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="r2")
grid_search.fit(X_train, y_train)

# Best model
best_knn = grid_search.best_estimator_

print("Best K (n_neighbors):", grid_search.best_params_["n_neighbors"])

predictions = best_knn.predict(X_test)

print("Final Model R² Score:", r2_score(y_test, predictions))
print("Final Model Mean Absolute Error:", mean_absolute_error(y_test, predictions))

#2. Stability test
param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring="r2")
grid.fit(X_train, y_train)

best_knn = grid.best_estimator_
baseline_preds = best_knn.predict(X_test)

print("Baseline Model Performance")
print("R2:", r2_score(y_test, baseline_preds))
print("MAE:", mean_absolute_error(y_test, baseline_preds))

# Add small random noise (±2%) to features
noise = np.random.normal(0, 0.02, X.shape)
X_noisy = X + X * noise

# Scale noisy data
X_noisy_scaled = scaler.fit_transform(X_noisy)

# Split again with same random state for fair comparison
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_noisy_scaled, y, test_size=0.3, random_state=42
)

grid_noisy = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring="r2")
grid_noisy.fit(X_train_n, y_train_n)

best_knn_noisy = grid_noisy.best_estimator_
noisy_preds = best_knn_noisy.predict(X_test_n)
print("\nModel Performance After Perturbation")
print("R2:", r2_score(y_test_n, noisy_preds))
print("MAE:", mean_absolute_error(y_test_n, noisy_preds))

difference = np.abs(baseline_preds - noisy_preds)

print("\nAverage Prediction Difference After Perturbation:", np.mean(difference))
print("Maximum Prediction Difference:", np.max(difference))

#3 
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("ORIGINAL MODEL PERFORMANCE ")
print("KNN R2:", r2_score(y_test, knn_preds))
print("KNN MAE:", mean_absolute_error(y_test, knn_preds))
print("Linear Regression R2:", r2_score(y_test, lr_preds))
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_preds))

noise = np.random.normal(0, 0.02, X.shape)
X_noisy = X + X * noise
X_noisy_scaled = scaler.fit_transform(X_noisy)

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_noisy_scaled, y, test_size=0.3, random_state=42
)

knn.fit(X_train_n, y_train_n)
knn_preds_noisy = knn.predict(X_test_n)

lr.fit(X_train_n, y_train_n)
lr_preds_noisy = lr.predict(X_test_n)

print("
=== STABILITY TEST RESULTS ===")
print("KNN Prediction Change (Avg):", np.mean(np.abs(knn_preds - knn_preds_noisy)))
print("Linear Regression Prediction Change (Avg):", np.mean(np.abs(lr_preds - lr_preds_noisy)))

feature_names = ["Advertising_Spend", "Price", "Competitor_Price"]
coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": lr.coef_
})

print("
=== LINEAR REGRESSION INTERPRETABILITY ===")
print(coefficients)