# TRONIX ESP32 Firmware

This directory contains the ESP32 firmware used for the TRONIX edge-based TinyML keyword spotting system.

## Main Firmware

`TRONIX_ESP32_Audio.ino` contains the embedded application used for audio acquisition, signal processing, TinyML inference, and wake-word event handling.

## Audio Acquisition

The firmware interfaces with the INMP441 digital MEMS microphone through the ESP32 I2S interface.

The audio configuration used by TRONIX is:

- Sampling rate: 16 kHz
- Audio: Mono
- Interface: I2S
- BCLK/SCK: GPIO26
- WS/LRCLK: GPIO25
- DOUT/SD: GPIO33

## Edge Processing

The firmware performs the main keyword-spotting operations locally on the ESP32. The processing pipeline includes RMS-based voice activity detection, MFCC feature extraction, and INT8 CNN inference.

The classifier recognizes four classes:

- Hey Leo
- Other Speech
- Similar Words
- Background Noise

## TinyML Model

The INT8 model is integrated into the firmware using the generated C/C++ model representation:

`tronix_94frame_int8.h`

The deployed model uses a 13 × 94 MFCC input representation and has a model size of approximately 34.39 KB.

## Wake-Word Event Handling

When the local classifier detects the target wake word, the firmware generates a wake event containing the relevant inference information. Non-wake audio is processed locally without continuous cloud audio transmission.

## Deployment

The firmware is intended to be compiled and uploaded to an ESP32 development board using the Arduino IDE or a compatible ESP32 development environment.
