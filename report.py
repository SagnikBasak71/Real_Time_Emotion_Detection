from pymongo import MongoClient
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import joblib
from utils import classify_emotion   # ✅ import rule

# ===============================
# 1) Connect to MongoDB
# ===============================
client = MongoClient("mongodb://localhost:27017/")
db = client["emotion_tracker"]
collection = db["emotions"]

# ===============================
# 2) Load trained Random Forest model
# ===============================
rf = joblib.load("rf_model.pkl")

# ===============================
# 3) Parameters
# ===============================
video_id = "VID001"
students = collection.distinct("student_id", {"video_id": video_id})

# ===============================
# 4) Generate report for each student
# ===============================
for student_id in students:
    records = list(collection.find(
        {"student_id": student_id, "video_id": video_id},
        {"_id": 0, "emotion": 1}
    ))
    
    emotions = [rec["emotion"] for rec in records]
    counts = Counter(emotions)

    df = pd.DataFrame.from_dict(counts, orient="index", columns=["count"])
    df.index.name = "emotion"
    df = df.sort_values(by="count", ascending=False)

    total = sum(counts.values())
    features = [
        (counts.get("happy", 0) / total) * 100 if total else 0,
        (counts.get("sad", 0) / total) * 100 if total else 0,
        (counts.get("angry", 0) / total) * 100 if total else 0,
        (counts.get("fear", 0) / total) * 100 if total else 0,
        (counts.get("neutral", 0) / total) * 100 if total else 0,
        (counts.get("surprise", 0) / total) * 100 if total else 0,
        (counts.get("disgust", 0) / total) * 100 if total else 0,
    ]
    features = np.array([features])

    rf_label = rf.predict(features)[0]

    # ✅ Rule-based classification
    percentages = {
        "happy": features[0][0], "sad": features[0][1], "angry": features[0][2],
        "fear": features[0][3], "neutral": features[0][4],
        "surprise": features[0][5], "disgust": features[0][6]
    }
    rule_label = classify_emotion(percentages)

    # ===============================
    # 5) Print report
    # ===============================
    print(f"\n📊 Report for Student {student_id}, Video {video_id}")
    print("Emotion counts:")
    print(df)
    print(f"➡️ Rule-based classification: {rule_label}")
    

    # ===============================
    # 6) Graphs
    # ===============================
    df.plot(kind="bar", legend=False)
    plt.title(f"Emotion Distribution - {student_id}")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df.plot(kind="pie", y="count", autopct="%1.1f%%", legend=False)
    plt.title(f"Emotion Breakdown - {student_id}")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
