import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load test dataset
df = pd.read_csv('test.csv')

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# Features for prediction
X_test = df[['Age', 'Sex', 'Fare']]

# Predict
predictions = model.predict(X_test)

# Get PassengerId if available, otherwise use default index
if 'PassengerId' in df.columns:
    ids = df['PassengerId']
else:
    ids = pd.Series(range(1, len(predictions) + 1))

# Create output DataFrame
output = pd.DataFrame({
    'PassengerId': ids,
    'PredictedSurvived': predictions
})

# Save predictions
output.to_csv('predictions.csv', index=False)
df.to_csv('test_with_predictions.csv', index=False)

print("✅ Predictions saved to predictions.csv")