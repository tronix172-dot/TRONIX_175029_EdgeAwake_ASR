# TRONIX Results

This directory contains the trained model, INT8 deployment model, and evaluation results for the TRONIX TinyML keyword spotting system.

## Model and Deployment Files

- `tronix_94frame_best.keras` — trained Keras model.
- `tronix_94frame_int8.tflite` — INT8 TensorFlow Lite model used for embedded deployment.
- `tronix_94frame_int8.h` — C/C++ representation of the INT8 model for ESP32 integration.
- `tronix_94frame_test_results.txt` — model evaluation and test results.

## Model Configuration

The TRONIX classifier uses a 13 × 94 MFCC feature representation extracted from 16 kHz mono audio.

The model performs four-class classification:

1. Hey Leo
2. Other Speech
3. Similar Words
4. Background Noise

## Validated Results

| Metric | Value |
|---|---:|
| Accuracy | 87.22% |
| Precision | 88.17% |
| Recall | 87.22% |
| F1-score | 87.28% |
| Model Size | 34.39 KB |

## Embedded Deployment

The INT8 model is deployed on an ESP32 for local keyword-spotting inference. The edge device performs audio processing and classification locally, with selective event communication after wake-word detection.
