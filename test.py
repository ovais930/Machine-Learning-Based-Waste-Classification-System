import tensorflow as tf

# Load model
model = tf.keras.models.load_model("models/waste-classifier-mini-project.h5")

# Load labels
with open("label.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Number of output classes from model
num_classes = model.output_shape[-1]

print("\n==============================")
print(" MODEL CLASS INFORMATION")
print("==============================")

print(f"Model Output Classes : {num_classes}")
print(f"Labels in label.txt  : {len(labels)}")

# Check if matching
if num_classes == len(labels):
    print("✔ The label list matches the model output.\n")
else:
    print("⚠ WARNING: Model output and label.txt mismatch!\n")

# Print all classes
print("Classes:")
for i, name in enumerate(labels):
    print(f"{i}: {name}")