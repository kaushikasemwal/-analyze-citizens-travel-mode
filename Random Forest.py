import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz

# Load the dataset
df = pd.read_csv(r'c:\Users\maitr\OneDrive\Documents\Kaushika\OT Project\table__84710ENG.csv')

# Clean column names
df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

# Encode the target variable
df['Travel_modes'] = LabelEncoder().fit_transform(df['Travel_modes'])

# Define numeric columns
numeric_cols = [
    'Distance_travelled__passenger_kilometres__',
    'Average_per_person_per_day_Trips__number_',
    'Average_per_person_per_year_Trips__number_'
]

# Convert to numeric and handle NaN values
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(0)  # Fill NaN values with 0 or use another method

# Split the data into features and target
X = df.drop('Travel_modes', axis=1)
y = df['Travel_modes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create a ColumnTransformer to handle both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Scale numeric columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical columns
    ]
)

# Create a pipeline with preprocessing and Random Forest Classifier
pipeline = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Scatter Plot with Best Fit Line (using training data for fit)
distance_travelled = X_train['Distance_travelled__passenger_kilometres__']  # Use X_train
travel_modes = y_train  # Use y_train

plt.figure(figsize=(10, 6)) # Adjust figure size for better visualization
plt.scatter(distance_travelled, travel_modes, color='blue', label='Data Points')

# Fit a best-fit line using numpy polyfit
coefficients = np.polyfit(distance_travelled, travel_modes, 1)  # Linear fit
best_fit_line = np.poly1d(coefficients)

# Plot the best-fit line
plt.plot(distance_travelled, best_fit_line(distance_travelled), color='red', label='Best Fit Line')

plt.xlabel('Distance Travelled (Passenger Kilometres)')
plt.ylabel('Travel Modes')
plt.title('Scatter Plot with Best Fit Line in Random Forest')
plt.legend()
plt.grid(True)  # Add grid for easier readability
plt.show()