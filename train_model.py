import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

# Load cleaned data
df = pd.read_csv('premier_league_cleaned.csv')

# Prepare data
X = df.drop('outcome', axis=1)
y = df['outcome']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Improved model with hyperparameter tuning
print("Training model with hyperparameter tuning...")

# Define parameter grid for RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create RandomForest classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Get best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = X.columns
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# Save both model and scaler
joblib.dump(best_model, 'premier_league_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nModel and scaler saved successfully!")
print(f"Training completed with {accuracy:.1%} accuracy")