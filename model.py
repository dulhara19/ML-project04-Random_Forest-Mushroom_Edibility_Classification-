from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
dataset = load_dataset("jlh/uci-mushrooms")

# Check available splits
#print(dataset)

df = pd.DataFrame(dataset ['train'])
#print(df)

# Initialize a LabelEncoder
encoder = LabelEncoder()

# Apply encoding to each column
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# Check the transformed dataset
print(df.head())

# Define target (first column) and features (remaining columns)
X = df.iloc[:, 1:]  # Features
y = df.iloc[:, 0]   # Target (poisonous or edible)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
