
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load the trained CNN
model = tf.keras.models.load_model("digitvision_cnn.keras")

# Load the unseen MNIST test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Match the CNN's 28 x 28 grayscale input
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# Predict digits
predictions = model.predict(x_test, batch_size=256)
y_pred = np.argmax(predictions, axis=1)

# Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=range(10))

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title(f"DigitVision CNN - Confusion Matrix ({accuracy*100:.2f}% Accuracy)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()
