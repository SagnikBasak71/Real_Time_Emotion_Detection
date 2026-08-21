import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TF info/warnings

import cv2
import time
import numpy as np
from collections import Counter
from tensorflow.keras.models import model_from_json
from pymongo import MongoClient
import csv

# ===============================
# 1) Auto-detect model files
# ===============================
model_dir = r"D:\emotion_tracker\emotion_model"
json_file, h5_file = None, None

for f in os.listdir(model_dir):
    p = os.path.join(model_dir, f)
    if f.lower().endswith(".json"):
        json_file = p
    elif f.lower().endswith(".h5"):
        h5_file = p

if not json_file:
    raise FileNotFoundError("❌ No JSON file found in " + model_dir)
if not h5_file:
    raise FileNotFoundError("❌ No H5 file found in " + model_dir)

print("✅ Found JSON:", json_file)
print("✅ Found H5:", h5_file)

# ===============================
# 2) Load model
# ===============================
with open(json_file, "r") as fp:
    model = model_from_json(fp.read())

model.load_weights(h5_file)
print("🎉 Model loaded successfully!")

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ===============================
# 3) MongoDB setup
# ===============================
client = MongoClient("mongodb://localhost:27017/")  # default local MongoDB
db = client["emotion_tracker"]
collection = db["emotions"]

student_id = "P1"
video_id = "VID001"

# ===============================
# 4) Haar Cascade (face ROI)
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("❌ Could not load Haar Cascade. Check OpenCV install.")

# ===============================
# 5) CSV setup
# ===============================
CSV_FILE = "emotion_features.csv"

def save_features_to_csv(features, label):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["happy", "sad", "angry", "fear", "neutral", "surprise", "disgust", "label"])
        writer.writerow(features + [label])

# ===============================
# 6) Webcam + 3s buffer
# ===============================
cap = cv2.VideoCapture(0)
WINDOW_DURATION = 3  # seconds
window_start = time.time()
emotion_buffer = []

def preprocess_roi(gray_img, box):
    x, y, w, h = box
    roi = gray_img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))  # shape: (1, 48, 48, 1)
    return roi

def pick_largest_face(faces):
    if len(faces) == 0:
        return None
    areas = [(w*h, i) for i, (_, _, w, h) in enumerate(faces)]
    _, idx = max(areas)
    return faces[idx]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    # Use largest face
    face_box = pick_largest_face(faces)

    if face_box is not None:
        x, y, w, h = face_box

        pad = int(0.1 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_box = (x1, y1, x2 - x1, y2 - y1)

        # Preprocess ROI and predict
        face_input = preprocess_roi(gray, face_box)
        probs = model.predict(face_input, verbose=0)[0]
        idx = int(np.argmax(probs))
        emotion = labels[idx]

        # Save to MongoDB
        collection.insert_one({
            "student_id": student_id,
            "video_id": video_id,
            "emotion": emotion,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # Add to buffer
        emotion_buffer.append(emotion)

        # Draw UI
        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            frame, f"{student_id}: {emotion}",
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
        )
    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection (MongoDB)", frame)

    # --------- 3-sec summary ----------
    if time.time() - window_start >= WINDOW_DURATION and len(emotion_buffer) > 0:
        counts = Counter(emotion_buffer)
        total = sum(counts.values())

        percentages = {
            e: (counts.get(e, 0) / total) * 100 for e in labels
        }

        positive_emotions = ["happy", "surprise","neutral"]
        positive_count = sum(counts.get(e, 0) for e in positive_emotions)
        positive_percent = (positive_count / total) * 100 if total > 0 else 0.0

        if positive_percent >= 80:
            final_label = "Positive"
        elif positive_percent >= 60:
            final_label = "Little Positive"
        else:
            final_label = "Negative"

        print(f"\n✅ 3-sec summary for {student_id}: {positive_percent:.2f}% → {final_label}")

        # save to CSV
        features = [
            percentages["happy"],
            percentages["sad"],
            percentages["angry"],
            percentages["fear"],
            percentages["neutral"],
            percentages["surprise"],
            percentages["disgust"]
        ]
        save_features_to_csv(features, final_label)
        print("📂 Saved row to CSV!")

        # reset
        window_start = time.time()
        emotion_buffer = []

    # quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
client.close()
