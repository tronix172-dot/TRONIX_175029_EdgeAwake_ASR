# TRONIX TinyML Model

This directory contains the TinyML model used for local keyword spotting in the TRONIX system.

## Model Input

The model uses MFCC features extracted from 16 kHz mono audio.

- Feature representation: 13 × 94 MFCC
- Audio sampling rate: 16 kHz
- Input type: MFCC feature matrix

## Model Architecture

TRONIX uses a compact convolutional neural network consisting of convolutional layers, batch normalization, pooling, global average pooling, and a dense classification layer.

The classifier recognizes four classes:

1. Hey Leo
2. Other Speech
3. Similar Words
4. Background Noise

## Quantization

The deployed model uses INT8 quantization to reduce the model footprint and support efficient inference on the ESP32.

The deployed model size is approximately 34.39 KB.

## Deployment

The generated C/C++ model representation is provided in:

`tronix_94frame_int8.h`

This model representation is included by the ESP32 firmware for local TinyML inference.

## Evaluation

The validated classification results reported for the TRONIX model are:

| Metric | Value |
|---|---:|
| Accuracy | 87.22% |
| Precision | 88.17% |
| Recall | 87.22% |
| F1-score | 87.28% |

The model performs inference locally on the ESP32, allowing non-wake audio to be processed at the edge without continuous cloud transmission.
