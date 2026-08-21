# build_dataset.py
import pandas as pd
from pymongo import MongoClient
from collections import Counter

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["emotion_tracker"]
collection = db["emotions"]

# Fetch all records (you can filter by student_id / video_id if needed)
records = list(collection.find())

# Convert to DataFrame
df = pd.DataFrame(records)

# Group by 60-second windows (you can adjust this logic)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["minute_block"] = df["timestamp"].dt.floor("60s")

features = []
for block, group in df.groupby("minute_block"):
    counts = Counter(group["emotion"])
    total = sum(counts.values())
    row = {
        "happy": counts.get("happy", 0) / total * 100 if total > 0 else 0,
        "sad": counts.get("sad", 0) / total * 100 if total > 0 else 0,
        "angry": counts.get("angry", 0) / total * 100 if total > 0 else 0,
        "fear": counts.get("fear", 0) / total * 100 if total > 0 else 0,
        "neutral": counts.get("neutral", 0) / total * 100 if total > 0 else 0,
        "surprise": counts.get("surprise", 0) / total * 100 if total > 0 else 0,
        "disgust": counts.get("disgust", 0) / total * 100 if total > 0 else 0,
        "label": "Unknown"  # <-- later you fill manually: Positive / Negative / Little Positive
    }
    features.append(row)

# Save to CSV
features_df = pd.DataFrame(features)
features_df.to_csv("emotion_features.csv", index=False)

print("✅ Dataset saved to emotion_features.csv")
