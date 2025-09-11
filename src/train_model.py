import pandas as pd
import numpy as np
from scipy.integrate import trapz
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # For saving the model

# Function to calculate the necessary features for each concentration
def extract_features(data):
    # Initialize a dictionary to store extracted features
    features = {}
    
    # Extract Peak Height (max current value)
    features['Peak_Height'] = data['Current'].max()
    
    # Extract Peak Potential (corresponding voltage to Peak Height)
    peak_idx = data['Current'].idxmax()
    features['Peak_Potential'] = data.loc[peak_idx, 'Voltage']
    
    # Calculate Area Under Curve using trapezoidal rule
    features['Area_Under_Curve'] = trapz(data['Current'], data['Voltage'])
    
    # Calculate Mean Current
    features['Mean_Current'] = data['Current'].mean()
    
    # Calculate Standard Deviation of Current
    features['Std_Current'] = data['Current'].std()
    
    # Calculate Skewness of Current
    features['Skew_Current'] = skew(data['Current'])
    
    return features

# Load the training dataset
train_data = pd.read_csv("data.csv")

# Extract features and target variable (Concentration)
X_train = train_data.drop(columns=['Concentration'])  # Features
y_train = train_data['Concentration']  # Target variable (Concentration)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on training data
y_train_pred = model.predict(X_train)

# Calculate evaluation metrics
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f"R² of the model: {r2:.4f}")

# Print evaluation details
# Plot Actual vs Predicted values


# Save the model after training
joblib.dump(model, 'rf_model.pkl')
print("Model saved successfully.")

# Save the model, features, and other important files
joblib.dump(model, 'rf_model.pkl')
