import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import joblib

# Load dataset
df = pd.read_csv("applicants.csv")

# Encode categorical column
encoder = LabelEncoder()
df["track"] = encoder.fit_transform(df["track"])

# Define features and target
X = df[["gpa", "skills_count", "prior_projects", "track"]]
y = df["shortlisted"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "model.joblib")
joblib.dump(encoder, "encoder.joblib")

print("Model trained and saved successfully!")
