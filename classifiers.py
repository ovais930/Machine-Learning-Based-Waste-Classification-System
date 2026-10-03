import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("models/waste-classifier-mini-project.h5")

# -----------------------------
# Load Labels From labels.txt
# -----------------------------
with open("Labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("Loaded classes:", labels)

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Prediction Smoothing
# -----------------------------
smooth_window = 7
pred_buffer = deque(maxlen=smooth_window)

# -----------------------------
# Start Camera
# -----------------------------
cap = cv2.VideoCapture(0)
print("Camera started... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess(frame)
    prediction = model.predict(processed, verbose=0)[0]

    # Find class
    class_id = np.argmax(prediction)
    confidence = prediction[class_id]
    pred_buffer.append(class_id)

    # Majority vote smoothing
    stable_class = max(set(pred_buffer), key=pred_buffer.count)
    stable_name = labels[stable_class]
    stable_conf = np.mean([prediction[stable_class] for _ in range(len(pred_buffer))])

    # Display text
    if stable_conf > 0.50:  # Avoid flicker
        text = f"{stable_name} ({stable_conf*100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = "Detecting..."
        color = (0, 0, 255)

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()