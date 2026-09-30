# TRONIX Backend

The TRONIX backend provides the server-side interface used for selective event logging from the ESP32 edge device.

## Purpose

TRONIX performs keyword-spotting inference locally on the ESP32. The backend is not used for continuous audio processing. Instead, communication is performed selectively when a relevant wake-word event is generated.

## Event Logging

A wake event can contain metadata associated with the local inference, such as:

- Timestamp
- Detected class
- Confidence
- RMS value
- Inference latency
- Device identifier
- Model identifier
- Event status

The system is designed so that non-wake audio does not require continuous cloud transmission.

## Privacy-Oriented Operation

The primary audio-processing and keyword-spotting operations are performed at the edge. The backend is therefore used for event-level logging rather than continuous streaming of microphone audio.

## Repository Status

The backend directory currently contains documentation describing the server-side role of the TRONIX system. Backend implementation files are not included in this repository because the deployed backend implementation is maintained separately.
