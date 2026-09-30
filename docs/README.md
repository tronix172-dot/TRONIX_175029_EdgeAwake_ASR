# TRONIX: Edge-Based TinyML Keyword Spotting System

TRONIX is an ESP32-based TinyML keyword spotting system using an INMP441 digital microphone.

The system processes audio locally using RMS-based VAD, MFCC feature extraction, and an INT8 CNN model. The target wake word is **"Hey Leo"**.

## Live Demo

[Open TRONIX Website Dashboard](https://tronix172-dot.github.io/TRONIX_175029_EdgeAwake_ASR/)

## Hardware

- ESP32
- INMP441 digital MEMS microphone
- 16 kHz mono audio
- I2S interface

## Machine Learning

- MFCC input: 13 × 94
- Model: Compact CNN
- Quantization: INT8
- Classes: 4
- Wake word: Hey Leo
- Model size: 34.39 KB

## Classes

| Class | Description |
|---|---|
| C0 | Hey Leo |
| C1 | Other Speech |
| C2 | Similar Words |
| C3 | Background Noise |

## Results

| Metric | Value |
|---|---:|
| Accuracy | 87.22% |
| Precision | 88.17% |
| Recall | 87.22% |
| F1-score | 87.28% |

## Repository Structure

```text
backend/     - Backend documentation
demo/        - Online demo link
docs/        - Project documentation
firmware/    - ESP32 firmware
hardware/    - Hardware information
ml/          - TinyML model
results/     - Models and test results
