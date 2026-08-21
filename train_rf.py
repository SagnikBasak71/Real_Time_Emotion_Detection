import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load your labeled dataset (CSV or MongoDB export)
df = pd.read_csv("emotion_features.csv")  # each row = 60 sec summary

# 2. Features (X) and target (y)
X = df[["happy", "sad", "angry", "fear", "neutral", "surprise", "disgust"]]
y = df["label"]   # Positive / Little Positive / Negative

# 3. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 4. Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("✅ Random Forest trained & saved!")
