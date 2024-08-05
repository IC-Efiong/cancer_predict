import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_model(data):
  """Trains a logistic regression model on the provided data.

  Args:
    data: A pandas DataFrame containing the features and target variable.

  Returns:
    A tuple containing the trained logistic regression model and the scaler used.
  """

  # Separate features and target variable
  X = data.drop('diagnosis', axis=1)
  y = data['diagnosis']

  # Scale the features
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

  # Train the logistic regression model
  model = LogisticRegression()
  model.fit(X_train, y_train)

  # Testing the model
  y_pred = model.predict(X_test)
  print('Model Accuracy:', accuracy_score(y_test, y_pred))
  print('Classification Report:', classification_report(y_test, y_pred))

  return model, scaler



def preprocess_data():
    """Reads, cleans, and preprocesses the dataset.

    Returns:
    A cleaned pandas DataFrame.
    """
    data = pd.read_csv("data/data.csv")
    data = data.drop(columns=['Unnamed: 32','id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    data = preprocess_data()
    model, scaler = train_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    



if __name__ == '__main__':
    main()