# TRONIX Demo

The TRONIX online demonstration provides a web dashboard for viewing the deployed edge-based TinyML keyword spotting system.

## Online Dashboard

[**Open TRONIX Website Dashboard →**](https://tronix172-dot.github.io/TRONIX_175029_EdgeAwake_ASR/)

The dashboard presents the edge-device status, TinyML model information, wake-word detection events, and system performance information.

## System

TRONIX uses an ESP32 with an INMP441 digital MEMS microphone. Audio is processed locally using RMS-based voice activity detection, MFCC feature extraction, and an INT8 CNN classifier.

The target wake word is **"Hey Leo"**.


## Online Demo

The TRONIX web dashboard provides a visual interface for observing the deployed system, including edge-device status, TinyML model information, wake-word detection events, and system performance information.

The live demonstration is available through the website link provided in this directory.

## Demonstration Features

The demonstration interface presents:

- Edge device information
- TinyML model status
- Wake-word detection status
- Event logging
- Inference information
- System performance information

## Project

TRONIX performs keyword spotting locally on an ESP32 using an INMP441 digital MEMS microphone, MFCC feature extraction, and an INT8 CNN model.

The target wake word is "Hey Leo".

For implementation details, refer to the `firmware/`, `hardware/`, `ml/`, `backend/`, and `results/` directories.
