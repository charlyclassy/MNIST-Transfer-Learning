# 🔢 DigitVision AI Studio — Multi-digit Handwriting Recognition

**[🚀 LAUNCH DIGITVISION AI](https://mnist-transfer-learning-mxwxdyytvvjsgw6adbt5mr.streamlit.app/)**

An enhanced Streamlit interface for your **existing MobileNetV2 handwritten-digit models**. It separates a handwritten number into individual digit images, classifies each image using the original saved CNN, and joins the predictions in reading order. Test inputs such as **20, 125, or 2026**.

## Files to retain from your original GitHub repository

**Do not delete these files:**
- `mnist_fine_tuned_model.keras`
- `mnist_transfer_model.keras`
- `confusion_matrix.png` (optional display)
- `sample_predictions.png` (optional display)
- Original training scripts and reports

This ZIP intentionally **does not contain copies of the trained model weights**. The app needs the two `.keras` files at repository root next to `app.py`.

## Install and run

Use **Python 3.12**.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Use

1. Upload a JPG or PNG containing a single handwritten number (possibly multiple digits).
2. View the detected purple segment boxes.
3. If adjacent digits are not split correctly, select **Manual cut positions** in the sidebar and enter comma-separated cut positions as percentages of the writing region (e.g. `33,66` for three digits).
4. Click **Recognise handwriting**.
5. Inspect individual predictions and confidence scores.

## Important technical notes

The model still recognises **individual digits 0–9**, as MNIST does. Multi-digit recognition here is implemented by **digit segmentation + per-digit classification**, not by training a new arbitrary-length sequence model. It works best on clearly separated handwriting; joined or overlapping characters may require manual correction. Each region is preprocessed to match the original model's expected input format internally.

The confidence values shown are model softmax scores, not guarantees of correctness. No new measured accuracy claim is made without separate testing.

## Deploy

Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore` and the `.streamlit` directory to your existing GitHub repository, while **keeping the trained model files**. Streamlit Community Cloud should be set to Python 3.12.
