import pandas as pd
import joblib

#loading our model 
model = joblib.load('random_forest_model.pkl')

#preprocess the test data
df = pd.read_csv('test.csv')

df.drop(['Ticket', 'Cabin', 'Embarked'], axis =1, inplace= True
)
print(df.columns)

df['Age'] = df['Age'].fillna(df[Age].median) #handle missing value in age column
#encoding sex column 
df['Sex'] = df['Sex'].map({'male':1, 'female':0})

#selecting same features to train model
X_test = df[['Age', 'Sex', 'Fare']]

#predict using the loaded test data

predictions = model.predict(X_test)

#save predictions to csv

passengerIds = df['PassengerId'] if 'PassengerId' in df.columns else range (1, len(predictions)+1)
output = pd.DataFrame(
    'PassengerId': passengerIds,
    'PredictedSurvived': predictions
)
output.to_csv('predictions.csv', index = False)

print("Predictions saved to predictions.csv")