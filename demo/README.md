# TRONIX Demo

This directory provides access to the online demonstration of the TRONIX edge-based TinyML keyword spotting system.

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
