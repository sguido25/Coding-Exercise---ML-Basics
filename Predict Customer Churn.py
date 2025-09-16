import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Generate sample customer data
# Data = https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

data = {
'age': [18, 22, 41, 47, 35, 53, 30, 47, 54, 36],
'monthly_usage_hours': [14, 28, 10, 12, 24, 14, 15, 11, 5, 4],
'purchase_amount': [598, 584, 757, 232, 533, 500, 574, 323, 687, 995],
'customer_service_calls': [4, 7, 2, 5, 9, 10, 9, 10, 9, 0],
'gender': ['Female', 'Female', 'male', 'male', 'Female', 'male', 'Female', 'Female',
'male','male'],
'churn': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0] # 1 = churned, 0 = not churned
}
df = pd.DataFrame(data)
# Features and target
X = df[['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls',
'gender']]
y = df['churn']
# Preprocessing: Scale numerical features and one-hot encode categorical features
preprocessor = ColumnTransformer(
transformers=[
('num', StandardScaler(), ['age', 'monthly_usage_hours', 'purchase_amount',
'customer_service_calls']),
('cat', OneHotEncoder(sparse_output=False), ['gender'])
])
# Create pipeline with preprocessing and model
model = Pipeline(steps=[
('preprocessor', preprocessor),
('classifier', LogisticRegression(random_state=42))
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Train model
model.fit(X_train, y_train)
# Predict churn probability for a new customer
new_customer = pd.DataFrame({
'age': [35],
'monthly_usage_hours': [20],
'purchase_amount': [150],
'customer_service_calls': [5],
'gender': ['male']
})
churn_probability = model.predict_proba(new_customer)[0][1] # Probability of churn
# Classify based on threshold (0.5)
threshold = 0.5
churn_prediction = 1 if churn_probability > threshold else 0
print(f"Churn Probability for new customer: {churn_probability:.2f}")
print(f"Churn Prediction (1 = churn, 0 = no churn): {churn_prediction}")
# Display model coefficients
feature_names = (model.named_steps['preprocessor']
.named_transformers_['cat']
.get_feature_names_out(['gender'])).tolist() + ['age',
'monthly_usage_hours', 'purchase_amount', 'customer_service_calls']
coefficients = model.named_steps['classifier'].coef_[0]
print("\nModel Coefficients:")
for feature, coef in zip(feature_names, coefficients):print(f"{feature}: {coef:.2f}")

