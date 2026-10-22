
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix

# Load your fine-tuned model
model = tf.keras.models.load_model(
    "mnist_fine_tuned_model.keras"
)

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Convert grayscale images to RGB
x_test = np.stack([x_test] * 3, axis=-1)

# Resize images to 32x32
x_test = tf.image.resize(x_test, [32, 32])

# Normalise for MobileNetV2
x_test = (x_test / 127.5) - 1.0

# Predict digit probabilities
predictions = model.predict(x_test, batch_size=32)

# Get predicted digit labels
y_pred = np.argmax(predictions, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("MNIST Transfer Learning Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
