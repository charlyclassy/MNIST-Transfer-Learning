"""Evaluate on untouched MNIST test split, after training."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='digitvision_cnn.keras')
    p.add_argument('--output', default='evaluation.json')
    args = p.parse_args()
    model = tf.keras.models.load_model(args.model, compile=False)
    (_, _), (x, y) = tf.keras.datasets.mnist.load_data()
    probs = model.predict(x.astype('float32')[..., None] / 255.0, batch_size=256, verbose=1)
    predictions = probs.argmax(axis=1)
    result = {
        'dataset': 'MNIST untouched test set', 'test_images': len(y),
        'accuracy': float(accuracy_score(y, predictions)),
        'classification_report': classification_report(y, predictions, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, predictions, labels=list(range(10))).tolist(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'MNIST test accuracy: {result["accuracy"]:.2%}')
    print(classification_report(y, predictions, digits=4, zero_division=0))
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
