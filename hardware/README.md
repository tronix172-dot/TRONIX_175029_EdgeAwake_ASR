# TRONIX Hardware

TRONIX uses an ESP32 microcontroller with an INMP441 digital MEMS microphone for local audio acquisition and TinyML keyword spotting.

## Hardware Components

- ESP32 development board
- INMP441 digital MEMS microphone

## INMP441–ESP32 Connections

| INMP441 Pin | ESP32 | Function |
|---|---|---|
| VDD | 3.3 V | Power |
| GND | GND | Ground |
| SCK/BCLK | GPIO26 | I2S clock |
| WS/LRCLK | GPIO25 | I2S word select |
| SD/DOUT | GPIO33 | Digital audio data |
| L/R | GND | Channel selection |

## Audio Configuration

The microphone provides a digital mono audio stream through the I2S interface. TRONIX uses a sampling rate of 16 kHz for keyword-spotting processing.

## Processing Pipeline

The captured audio is processed locally on the ESP32 using RMS-based voice activity detection, MFCC feature extraction, and an INT8 CNN classifier. The target wake word is "Hey Leo".

## Deployment

The TinyML model is deployed on the ESP32 for local inference. Cloud communication is used selectively for wake-event logging rather than continuously transmitting audio.
