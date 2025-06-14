import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report 

#loading dataset using pandas
df = pd.read_csv('train.csv')

#cleaning data 
df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df.columns)

#filling missing data with median value in age column 

df['Age'] = df ['Age'].fillna(df['Age'].median())

#encoding sex column 
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']) 

#Features selection
features = ['Age', 'Sex', 'Fare']

X=df[features]
y= df['Survived']

#train model

model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X, y)

#evaluating on training set

y_pred = model.predict(X)

print("Training Accuracy:", accuracy_score(y, y_pred))
print ("classification Report:\n", classification_report(y, y_pred))


# Save to a text file

accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)
with open("model_evaluation.txt", "w") as f:
    f.write(f"Training Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Evaluation results saved to model_evaluation.txt")

import joblib

joblib.dump(model, 'random_forest_model.pkl')
print("Model saved as random_forest_model.pkl")
