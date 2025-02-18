import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# Load the trained model
model = joblib.load('saved_model.pkl')


# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# The iris dataset's default feature names:
default_feature_names = iris.feature_names  


rename_map = {
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width'
}


# Create a DataFrame and rename the columns
X_df = pd.DataFrame(X, columns=default_feature_names)
X_df.rename(columns=rename_map, inplace=True)


# If your training model was trained on string labels (e.g., "Iris-setosa"),
# you need to convert the numeric targets to string labels.
# Assuming the mapping is as follows:
target_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
y_str = [target_mapping[val] for val in y]


# Split the test data (here we use a test split for evaluation)
_, X_test, _, y_test = train_test_split(X_df, y_str, test_size=0.2, random_state=42)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)