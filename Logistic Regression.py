import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create a ColumnTransformer to handle both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Scale numeric columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical columns
    ]
)

# Create a pipeline with preprocessing and logistic regression
pipeline = make_pipeline(preprocessor, LogisticRegression())

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))