# 🧠 DigitVision AI — MNIST Transfer Learning

A redesigned Streamlit interface for the **original** trained MobileNetV2 handwritten digit recognition models.

## 🚀 [Launch the live DigitVision AI app](https://mnist-transfer-learning-mxwxdyytvvjsgw6adbt5mr.streamlit.app/)

## Important: Preserve existing files

**Do not delete your trained model files.** The new `app.py` loads:
- `mnist_fine_tuned_model.keras` (default)
- `mnist_transfer_model.keras` (optional baseline)

It also displays your existing `confusion_matrix.png` and `sample_predictions.png`.

The package provides replacement `app.py`, `requirements.txt`, `.streamlit/config.toml`, and `README.md`. Keep all existing training scripts, model files, report and images in GitHub.

## How it works

Uploaded images are converted to grayscale, resized to **32×32**, repeated into **3 RGB channels**, and normalized to **[−1, 1]**, matching the existing deployed `app.py` preprocessing. Toggle inversion if you have dark handwriting on a light background.

## Deploy

Commit the four replacement files to your **existing** GitHub repository and keep its original `.keras` model files. Redeploy from `main` using `app.py` and **Python 3.12** in Streamlit Cloud.

## Academic note

Original training code uses MNIST test data for validation during training and also for final evaluation. For rigorous future performance claims, train with a separate validation set and reserve the test split until the very end.
