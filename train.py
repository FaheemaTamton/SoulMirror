
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

#  Paths 
DATA_FILE = "data/16p.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

#  Load dataset
data = pd.read_csv(DATA_FILE, encoding="ISO-8859-1")
# Strip whitespace from column names
data.columns = data.columns.str.strip()
print("Columns in dataset:", data.columns.tolist())

#  Features and label
LABEL_COL = "Personality"
X = data.drop([LABEL_COL, "Response Id"], axis=1)  # drop label + Response Id
y = data[LABEL_COL]

#  Train a quick RandomForest to get feature importance
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

#  Select top 10 questions 
importances = pd.DataFrame({
    'question': X.columns,
    'importance': rf.feature_importances_
})
importances = importances.sort_values(by='importance', ascending=False)
top10_questions = importances['question'].head(10).tolist()
print("Top 10 questions selected:", top10_questions)

#  Prepare reduced dataset
X_reduced = X[top10_questions]

#  Scale data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

#  Train/test split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  Train final model 
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

#  Save model and scaler 
joblib.dump(model, os.path.join(MODEL_DIR, "personality_model_10.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_10.pkl"))

print("Training complete! Model and scaler saved for 10 questions.")
