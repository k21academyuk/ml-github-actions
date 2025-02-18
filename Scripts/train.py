import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Added for saving the model

# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/iris.csv")  # Adjust path

# Debugging: Print the current working directory and expected CSV path
print("Current Working Directory:", os.getcwd())
print("Expected Path for iris.csv:", csv_path)

# Load dataset
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Error: The file 'iris.csv' was not found. Check the file path.")
    exit(1)  # Exit script if the file is missing

# Prepare dataset
X = data.drop('species', axis=1)
y = data['species']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
model_filename = os.path.join(script_dir, "../saved_model.pkl")  # Save outside Scripts folder

# Save the model
joblib.dump(model, model_filename)

print(f"Training complete. Accuracy: {accuracy}")
