import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the fine-tuned model
model = tf.keras.models.load_model("mnist_fine_tuned_model.keras")

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Keep original images for display
x_display = x_test.copy()

# Convert grayscale to RGB
x_test = np.stack([x_test] * 3, axis=-1)

# Resize to 32x32
x_test = tf.image.resize(x_test, [32, 32])

# Preprocess for MobileNetV2
x_test = (x_test / 127.5) - 1.0

# Predict classes
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)

# Show first 10 test images
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_display[i], cmap="gray")
    plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=300, bbox_inches="tight")
plt.show()