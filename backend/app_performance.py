# ============================================================
# TRONIX - Edge AI Voice Security
# Flask Backend
# ============================================================
import json
import os
from datetime import datetime
from flask import Flask, jsonify, render_template
import serial
import threading
import time
import struct
import os
import json
from datetime import datetime

import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
# ------------------------------------------------------------
# TFLite
# ------------------------------------------------------------

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        Interpreter = None


# ============================================================
# FLASK
# ============================================================

import os
from flask import Flask, render_template

BASE_DIR = r"C:\Tronix_Web"

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
EVENTS_FILE = os.path.join("data", "events.json")


# ============================================================
# FIREBASE / CLOUD FIRESTORE
# ============================================================

FIREBASE_KEY = r"C:\Tronix_Web\firebase-key.json"

try:

    if not firebase_admin._apps:

        firebase_cred = credentials.Certificate(
            FIREBASE_KEY
        )

        firebase_admin.initialize_app(
            firebase_cred
        )

    db = firestore.client()

    FIREBASE_READY = True

    print("FIREBASE: Connected successfully.")

except Exception as e:

    db = None

    FIREBASE_READY = False

    print(
        "FIREBASE CONNECTION ERROR:",
        e
    )


def save_event(confidence, rms, latency_ms):
    """Save a successful Hey Leo detection to events.json."""
    try:
        os.makedirs("data", exist_ok=True)

        # Create file if it does not exist
        if not os.path.exists(EVENTS_FILE):
            with open(EVENTS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            events = json.load(f)

        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": "HEY LEO DETECTED",
            "confidence": round(float(confidence) * 100, 2),
            "rms": round(float(rms), 2),
            "latency_ms": round(float(latency_ms), 2),
            "device": "ESP32",
            "model": "TRONIX INT8",
            "audio_uploaded": False,
            "status": "SUCCESS"
        }
                # ----------------------------------------------------
        # SAVE TO FIREBASE CLOUD FIRESTORE
        # ----------------------------------------------------

        if db is not None:

            try:

                cloud_event = dict(event)

                cloud_event["cloud_logged_at"] = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                db.collection(
                    "wake_events"
                ).add(
                    cloud_event
                )

                print(
                    "CLOUD EVENT SAVED: "
                    "Firebase Firestore"
                )

            except Exception as cloud_error:

                print(
                    "CLOUD EVENT ERROR:",
                    cloud_error
                )

        events.insert(0, event)

        # Keep latest 100 events
        events = events[:100]

        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)

        print("EVENT LOG SAVED:", event)

    except Exception as e:
        print("EVENT LOG ERROR:", e)


def load_saved_events():
    """Load persistent wake events into the in-memory event log."""
    global event_logs

    try:
        if not os.path.exists(EVENTS_FILE):
            return

        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        if not isinstance(saved, list):
            return

        with state_lock:
            event_logs = []
            for item in saved[:100]:
                event_logs.append({
                    "time": item.get("timestamp", ""),
                    "event": item.get("event", "WAKE DETECTED"),
                    "prediction": "C0_Wake_Word",
                    "confidence": float(item.get("confidence", 0)),
                    "rms": float(item.get("rms", 0)),
                    "latency_ms": float(item.get("latency_ms", 0))
                })

        print("Loaded", len(event_logs), "saved event(s).")

    except Exception as e:
        print("EVENT LOG LOAD ERROR:", e)

# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM9"
BAUD = 921600

SAMPLE_RATE = 16000
DURATION_SECONDS = 3
NUM_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
AUDIO_BYTES = NUM_SAMPLES * 2

MODEL_PATH = (
    r"C:\Leo_Dataset\ML_Results\Reduced94"
    r"\tronix_94frame_int8.tflite"
)

LABELS = [
    "C0_Wake_Word",
    "C1_Other_Speech",
    "C2_Similar_Words",
    "C3_Background_Noise"
]

DISPLAY_LABELS = {
    "C0_Wake_Word": "Wake Word",
    "C1_Other_Speech": "Other Speech",
    "C2_Similar_Words": "Similar Words",
    "C3_Background_Noise": "Background Noise"
}


# ============================================================
# MODEL SETTINGS
# ============================================================

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 512

# Your working model quantization
INPUT_SCALE = 4.038010120391846
INPUT_ZERO_POINT = 75

OUTPUT_SCALE = 0.00390625
OUTPUT_ZERO_POINT = -128


# ============================================================
# DECISION THRESHOLDS
# These are the working values you were using.
# ============================================================

RMS_THRESHOLD = 450.0

WAKE_THRESHOLD = 0.30
SPEECH_THRESHOLD = 0.18


# ============================================================
# GLOBAL STATE
# ============================================================

serial_connection = None

worker_thread = None
stop_event = threading.Event()

state_lock = threading.Lock()

system_running = False
esp32_connected = False
current_state = "STANDBY"

last_prediction = "Waiting"
last_confidence = 0.0
last_rms = 0.0

last_probabilities = {
    "C0": 0.0,
    "C1": 0.0,
    "C2": 0.0,
    "C3": 0.0
}

last_detection_time = None
last_latency_ms = 0.0

total_cycles = 0
wake_count = 0
last_detection_timestamp = 0.0
WAKE_COOLDOWN_SECONDS = 3.0
last_wake_event_time = 0.0
event_logs = []

last_error = ""

# ============================================================
# ESP32 PERFORMANCE STATE
# ============================================================
# Last verified value from the ESP32 performance monitor.
# This is intentionally kept separate from the ML state.
last_performance = {
    "cpu_percent": 0.46,
    "free_heap": 221856,
    "min_free_heap": 199932,
    "largest_free_block": 110580,
    "total_heap": 278004,
    "timestamp": "Last verified",
    "source": "LAST VERIFIED",
}

last_performance_update = 0.0
PERFORMANCE_LIVE_TIMEOUT = 10.0

# Restore previously saved wake events for the website.
load_saved_events()


# ============================================================
# MODEL INITIALIZATION
# ============================================================

interpreter = None
input_details = None
output_details = None

if Interpreter is not None and os.path.exists(MODEL_PATH):

    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("==============================================")
        print("TRONIX MODEL LOADED")
        print("Model:", MODEL_PATH)
        print("Input:", input_details[0]["shape"])
        print("Output:", output_details[0]["shape"])
        print("==============================================")

    except Exception as e:
        print("MODEL LOAD ERROR:", e)
        interpreter = None

else:
    if Interpreter is None:
        print("ERROR: TensorFlow / TFLite runtime not available.")

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found:")
        print(MODEL_PATH)


# ============================================================
# LIBROSA
# ============================================================

try:
    import librosa
except Exception:
    librosa = None
    print("ERROR: librosa is not installed.")


# ============================================================
# EVENT LOG
# ============================================================

def add_log(
    event_type,
    prediction,
    confidence,
    rms=0.0,
    latency=0.0
):

    global event_logs

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log = {
        "time": timestamp,
        "event": event_type,
        "prediction": prediction,
        "confidence": round(float(confidence) * 100, 2),
        "rms": round(float(rms), 2),
        "latency_ms": round(float(latency), 2)
    }

    with state_lock:

        event_logs.insert(0, log)

        # Keep latest 100 events
        if len(event_logs) > 100:
            event_logs = event_logs[:100]


# ============================================================
# SERIAL HELPERS
# ============================================================

def safe_close_serial():

    global serial_connection
    global esp32_connected

    try:

        if serial_connection is not None:

            if serial_connection.is_open:
                serial_connection.close()

    except Exception:
        pass

    serial_connection = None
    esp32_connected = False


def connect_esp32():

    global serial_connection
    global esp32_connected
    global last_error

    try:

        safe_close_serial()

        print()
        print("Connecting to ESP32...")
        print("Port:", PORT)
        print("Baud:", BAUD)

        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD,
            timeout=0.20,
            write_timeout=1
        )

        time.sleep(0.2)

        # Clear old serial data safely
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        try:
            ser.reset_output_buffer()
        except Exception:
            pass

        serial_connection = ser
        esp32_connected = True
        last_error = ""

        print("ESP32 connected.")

        return True

    except Exception as e:

        serial_connection = None
        esp32_connected = False

        last_error = str(e)

        print("ESP32 CONNECTION ERROR:", e)

        return False


def send_command(command):

    global serial_connection

    try:

        if serial_connection is None:
            return False

        if not serial_connection.is_open:
            return False

        serial_connection.write(command.encode())
        serial_connection.flush()

        print("Command sent:", command.strip())

        return True

    except Exception as e:

        print("SEND ERROR:", e)

        return False


# ============================================================
# SERIAL LINE READER
# ============================================================

def parse_performance_line(text):
    """
    Parse the ESP32 performance line:

        PERF,cpu,free_heap,min_free_heap,largest_block,total_heap

    Example:
        PERF,0.46,221856,199932,110580,278004
    """
    global last_performance
    global last_performance_update

    try:
        if not text.startswith("PERF,"):
            return False

        parts = text.strip().split(",")

        if len(parts) != 6:
            return False

        cpu_percent = float(parts[1])
        free_heap = int(parts[2])
        min_free_heap = int(parts[3])
        largest_free_block = int(parts[4])
        total_heap = int(parts[5])

        # Basic sanity checks so random serial text cannot become
        # a fake performance reading.
        if not (0.0 <= cpu_percent <= 100.0):
            return False

        if min_free_heap < 0 or free_heap < 0 or total_heap <= 0:
            return False

        with state_lock:
            last_performance = {
                "cpu_percent": round(cpu_percent, 2),
                "free_heap": free_heap,
                "min_free_heap": min_free_heap,
                "largest_free_block": largest_free_block,
                "total_heap": total_heap,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "source": "LIVE ESP32",
            }

        last_performance_update = time.time()

        print(
            "PERFORMANCE UPDATE:",
            last_performance
        )

        return True

    except Exception as e:
        print("PERFORMANCE PARSE ERROR:", e)
        return False


def read_line(timeout_seconds=10):

    global serial_connection

    start = time.time()

    while not stop_event.is_set():

        if serial_connection is None:
            return None

        try:

            line = serial_connection.readline()

            if line:

                try:
                    text = line.decode(
                        "utf-8",
                        errors="ignore"
                    ).strip()

                except Exception:
                    text = ""

                if text:
                    # Performance reports are ordinary text lines.
                    # Store them and still return the line so existing
                    # handshake/status logic continues to work.
                    parse_performance_line(text)
                    return text

        except Exception as e:

            print("SERIAL READ ERROR:", e)
            return None

        if time.time() - start > timeout_seconds:
            return None

    return None


# ============================================================
# READ EXACT AUDIO BYTES
# ============================================================

def read_exact_audio(num_bytes):

    global serial_connection

    buffer = bytearray()

    while len(buffer) < num_bytes:

        if stop_event.is_set():
            return None

        if serial_connection is None:
            return None

        try:

            remaining = num_bytes - len(buffer)

            chunk = serial_connection.read(
                min(4096, remaining)
            )

            if chunk:
                buffer.extend(chunk)

            else:
                time.sleep(0.002)

        except Exception as e:

            print("AUDIO READ ERROR:", e)
            return None

    return bytes(buffer)


# ============================================================
# AUDIO → MFCC
# ============================================================

def extract_mfcc(audio):

    if librosa is None:
        raise RuntimeError("librosa is not installed.")

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Force exactly 3 seconds
    # --------------------------------------------------------

    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )

    elif len(audio) > NUM_SAMPLES:

        audio = audio[:NUM_SAMPLES]

    # --------------------------------------------------------
    # Peak normalization
    # Same preprocessing used by working Python model.
    # --------------------------------------------------------

    peak = np.max(np.abs(audio))

    if peak > 0:

        audio = audio / peak

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # Ensure 94 frames
    # --------------------------------------------------------

    if mfcc.shape[1] < 94:

        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                (0, 94 - mfcc.shape[1])
            ),
            mode="constant"
        )

    elif mfcc.shape[1] > 94:

        mfcc = mfcc[:, :94]

    return mfcc.astype(np.float32)


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_predict(audio):

    if interpreter is None:
        raise RuntimeError("TFLite model is not loaded.")

    mfcc = extract_mfcc(audio)

    # Shape = 13 x 94
    input_data = mfcc.reshape(
        1,
        N_MFCC,
        94,
        1
    )

    # --------------------------------------------------------
    # INT8 quantization
    # --------------------------------------------------------

    quantized = np.round(
        input_data / INPUT_SCALE
        + INPUT_ZERO_POINT
    )

    quantized = np.clip(
        quantized,
        -128,
        127
    ).astype(np.int8)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_details[0]["index"],
        quantized
    )

    interpreter.invoke()

    raw_output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # --------------------------------------------------------
    # Dequantize
    # --------------------------------------------------------

    output = (
        raw_output.astype(np.float32)
        - OUTPUT_ZERO_POINT
    ) * OUTPUT_SCALE

    output = output.flatten()

    # Safety normalization
    output = np.clip(output, 0.0, 1.0)

    total = np.sum(output)

    if total > 0:
        output = output / total

    return output


# ============================================================
# FINAL DECISION
# ============================================================

def final_decision(probabilities, rms):

    c0 = float(probabilities[0])
    c1 = float(probabilities[1])
    c2 = float(probabilities[2])
    c3 = float(probabilities[3])

    # --------------------------------------------------------
    # Very quiet signal
    # --------------------------------------------------------

    if rms < RMS_THRESHOLD:

        return 3

    # --------------------------------------------------------
    # Wake word
    # --------------------------------------------------------

    speech_max = max(c0, c1, c2)

    if (
        c0 >= WAKE_THRESHOLD
        and c0 >= c1
        and c0 >= c2
    ):

        return 0

    # --------------------------------------------------------
    # Other speech
    # --------------------------------------------------------

    if (
        c1 >= SPEECH_THRESHOLD
        and c1 >= c2
        and c1 >= c0
    ):

        return 1

    # --------------------------------------------------------
    # Similar words
    # --------------------------------------------------------

    if (
        c2 >= SPEECH_THRESHOLD
        and c2 >= c1
        and c2 >= c0
    ):

        return 2

    # --------------------------------------------------------
    # If there is reasonable speech confidence
    # --------------------------------------------------------

    if speech_max >= 0.12:

        return int(
            np.argmax(
                probabilities[:3]
            )
        )

    # Otherwise background
    return 3


# ============================================================
# PROCESS ONE AUDIO RECORDING
# ============================================================

def process_audio(audio_bytes):

    global last_prediction
    global last_confidence
    global last_rms
    global last_probabilities
    global last_detection_time
    global last_latency_ms
    global total_cycles
    global wake_count
    global current_state
    global last_detection_timestamp

    if audio_bytes is None:
        return None

    if len(audio_bytes) != AUDIO_BYTES:

        print(
            "Wrong audio size:",
            len(audio_bytes),
            "expected:",
            AUDIO_BYTES
        )

        return None

    # --------------------------------------------------------
    # Convert raw PCM16
    # --------------------------------------------------------

    audio = np.frombuffer(
        audio_bytes,
        dtype=np.int16
    ).astype(np.float32)

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    last_rms = rms

    # --------------------------------------------------------
    # Start timing
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    probabilities = model_predict(audio)

    inference_end = time.perf_counter()

    latency = (
        inference_end - inference_start
    ) * 1000.0

    prediction_index = final_decision(
        probabilities,
        rms
    )

    prediction_name = LABELS[prediction_index]

    confidence = float(
        probabilities[prediction_index]
    )

    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    with state_lock:

        last_prediction = prediction_name
        last_confidence = confidence
        last_latency_ms = latency

        last_probabilities = {
            "C0": round(
                float(probabilities[0]),
                6
            ),
            "C1": round(
                float(probabilities[1]),
                6
            ),
            "C2": round(
                float(probabilities[2]),
                6
            ),
            "C3": round(
                float(probabilities[3]),
                6
            )
        }

        total_cycles += 1

    print()
    print("=" * 55)
    print("TRONIX RESULT")

    print(
        f"C0 Wake Word: "
        f"{probabilities[0] * 100:.2f}%"
    )

    print(
        f"C1 Other Speech: "
        f"{probabilities[1] * 100:.2f}%"
    )

    print(
        f"C2 Similar Words: "
        f"{probabilities[2] * 100:.2f}%"
    )

    print(
        f"C3 Background: "
        f"{probabilities[3] * 100:.2f}%"
    )

    print(
        "Final:",
        prediction_name
    )

    print(
        f"RMS: {rms:.2f}"
    )

    print(
        f"Latency: {latency:.2f} ms"
    )

    print("=" * 55)

    # --------------------------------------------------------
    # Wake event
    # --------------------------------------------------------

    if prediction_index == 0:

        now = time.time()

        if now - last_detection_timestamp < 3.0:

            print("Duplicate HEY LEO ignored")

            current_state = "LISTENING"

            return prediction_index

        last_detection_timestamp = now

        wake_count += 1

        current_state = "WAKE DETECTED"

        last_detection_time = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        print()
        print(">>> HEY LEO DETECTED <<<")
        print()

        add_log(
            "WAKE DETECTED",
            prediction_name,
            confidence,
            rms,
            latency
        )

        save_event(
            confidence=confidence,
            rms=rms,
            latency_ms=latency
        )

        time.sleep(0.3)

        current_state = "LISTENING"
    else:

        current_state = "LISTENING"

        # Log only meaningful non-wake classifications
        add_log(
            "CLASSIFICATION",
            prediction_name,
            confidence,
            rms,
            latency
        )

    return prediction_index


# ============================================================
# WAIT FOR ESP32 READY
# ============================================================

def wait_for_ready():

    print("Waiting for ESP32 READY...")

    deadline = time.time() + 10

    while time.time() < deadline:

        if stop_event.is_set():
            return False

        line = read_line(
            timeout_seconds=1
        )

        if line is None:
            continue

        print("ESP32:", line)

        if (
            "ESP32_READY" in line
            or line == "READY"
            or "READY" in line
        ):
            return True

    return False


# ============================================================
# CONTINUOUS AUDIO STREAM
# ============================================================

def continuous_audio_worker():
    """
    Continuously receive PCM audio from ESP32.

    ESP32 sends:
        STREAM_START
        <continuous binary PCM16 audio>
        ...
        STREAM_STOP

    Python maintains a rolling 3-second buffer.

    Once 3 seconds are available:
        -> run TinyML
        -> keep the latest 2.5 seconds
        -> receive the next 0.5 seconds
        -> run again

    This removes the recording gaps between 3-second windows.
    """

    global current_state
    global system_running
    global worker_thread

    print()
    print("==============================================")
    print("TRONIX CONTINUOUS AUDIO WORKER")
    print("==============================================")

    current_state = "CONNECTING"

    # --------------------------------------------------------
    # CONNECT ESP32
    # --------------------------------------------------------

    if not connect_esp32():

        current_state = "CONNECTION ERROR"
        system_running = False

        print("TRONIX: ESP32 connection failed.")

        return

    # --------------------------------------------------------
    # HANDSHAKE
    # --------------------------------------------------------

    send_command("H")

    ready = wait_for_ready()

    if not ready:

        print(
            "TRONIX: ESP32 did not become ready."
        )

        current_state = "CONNECTION ERROR"
        system_running = False

        safe_close_serial()

        return

    # --------------------------------------------------------
    # START CONTINUOUS STREAM
    # --------------------------------------------------------

    print()
    print("Starting continuous microphone listening...")

    send_command("C")

    # --------------------------------------------------------
    # WAIT FOR STREAM_START
    # --------------------------------------------------------

    stream_started = False

    deadline = time.time() + 5

    while time.time() < deadline:

        if stop_event.is_set():
            break

        line = read_line(
            timeout_seconds=1
        )

        if line is None:
            continue

        print(
            "ESP32:",
            line
        )

        if "STREAM_START" in line:

            stream_started = True
            break

    if not stream_started:

        print(
            "ERROR: Continuous audio stream did not start."
        )

        current_state = "CONNECTION ERROR"
        system_running = False

        safe_close_serial()

        return

    # --------------------------------------------------------
    # LISTENING
    # --------------------------------------------------------

    current_state = "LISTENING"

    print()
    print("==============================================")
    print("TRONIX IS NOW LISTENING CONTINUOUSLY")
    print("==============================================")
    print("Microphone is active.")
    print("No recording gaps.")
    print("Waiting for first 3-second window...")
    print()

    # --------------------------------------------------------
    # ROLLING BUFFER
    # --------------------------------------------------------

    rolling_buffer = bytearray()

    WINDOW_BYTES = AUDIO_BYTES

    # Advance every 0.5 second.
    STEP_SAMPLES = int(
        SAMPLE_RATE * 0.5
    )

    STEP_BYTES = STEP_SAMPLES * 2
    
    INFERENCE_INTERVAL = 0.5
    next_inference_time = 0.0

    # --------------------------------------------------------
    # RECEIVE CONTINUOUS AUDIO
    # --------------------------------------------------------

    while not stop_event.is_set():

        try:

            if serial_connection is None:
                break

            # ------------------------------------------------
            # Read available binary audio
            # ------------------------------------------------

            waiting = serial_connection.in_waiting

            if waiting > 0:

                chunk = serial_connection.read(
                    min(waiting, 8192)
                )

                if chunk:

                    rolling_buffer.extend(
                        chunk
                    )

            else:

                # Keep CPU usage low while waiting
                time.sleep(0.005)

            # ------------------------------------------------
            # Wait until 3 seconds of audio are available
            # ------------------------------------------------

            if len(rolling_buffer) < WINDOW_BYTES:

                current_state = "LISTENING"

                continue

            # ------------------------------------------------
            # Take latest 3-second window
            # ------------------------------------------------

            audio_window = bytes(
                rolling_buffer[
                    -WINDOW_BYTES:
                ]
            )

            # ------------------------------------------------
            # Process current window
            # ------------------------------------------------

            current_state = "PROCESSING"

            try:

                process_audio(
                    audio_window
                )

            except Exception as e:

                print()
                print(
                    "CONTINUOUS INFERENCE ERROR:",
                    e
                )

                current_state = "ERROR"

            # ------------------------------------------------
            # Advance rolling window by 0.5 sec
            #
            # Keep 2.5 seconds and wait for the next
            # 0.5 seconds of fresh audio.
            # ------------------------------------------------

            if len(rolling_buffer) >= STEP_BYTES:

                rolling_buffer = bytearray(
                    rolling_buffer[
                        STEP_BYTES:
                    ]
                )

            else:

                rolling_buffer.clear()

            # ------------------------------------------------
            # Return to listening immediately
            # ------------------------------------------------

            if not stop_event.is_set():

                current_state = "LISTENING"

        except Exception as e:

            print()
            print(
                "CONTINUOUS AUDIO ERROR:",
                e
            )

            current_state = "ERROR"

            time.sleep(0.1)

    # ========================================================
    # STOP CONTINUOUS STREAM
    # ========================================================

    print()
    print(
        "Stopping continuous microphone stream..."
    )

    try:

        send_command("S")

    except Exception:
        pass

    # --------------------------------------------------------
    # Give ESP32 a moment to send STREAM_STOP
    # --------------------------------------------------------

    time.sleep(0.1)

    print(
        "TRONIX continuous audio worker stopped."
    )

    safe_close_serial()

    current_state = "STANDBY"
    system_running = False

    worker_thread = None


# ============================================================
# DETECTION WORKER
# ============================================================

def detection_worker():

    return continuous_audio_worker()

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    if not connect_esp32():

        current_state = "CONNECTION ERROR"
        system_running = False

        print(
            "TRONIX: ESP32 connection failed."
        )

        return

    # --------------------------------------------------------
    # Handshake
    # --------------------------------------------------------

    send_command("H")

    ready = wait_for_ready()

    if not ready:

        print(
            "TRONIX: ESP32 did not become ready."
        )

        current_state = "CONNECTION ERROR"
        system_running = False

        safe_close_serial()

        return

    # --------------------------------------------------------
    # Continuous listening
    # --------------------------------------------------------

    current_state = "LISTENING"

    print()
    print("TRONIX IS NOW LISTENING")
    print("Press STOP SYSTEM from website to stop.")
    print()

    while not stop_event.is_set():

        # ----------------------------------------------------
        # Ask ESP32 for recording
        # ----------------------------------------------------

        if not send_command("R"):
            break

        # ----------------------------------------------------
        # Wait for RECORDING_START
        # ----------------------------------------------------

        recording_started = False

        while not stop_event.is_set():

            line = read_line(
                timeout_seconds=5
            )

            if line is None:
                break

            print(
                "ESP32:",
                line
            )

            if "RECORDING_START" in line:

                recording_started = True
                current_state = "CAPTURING"
                break

        if stop_event.is_set():
            break

        if not recording_started:

            print(
                "Recording did not start."
            )

            time.sleep(0.2)
            continue

        # ----------------------------------------------------
        # Wait for RECORDING_DONE
        # ----------------------------------------------------

        recording_done = False

        while not stop_event.is_set():

            line = read_line(
                timeout_seconds=5
            )

            if line is None:
                break

            print(
                "ESP32:",
                line
            )

            if "RECORDING_DONE" in line:

                recording_done = True
                break

        if stop_event.is_set():
            break

        if not recording_done:

            print(
                "Recording did not finish."
            )

            continue

        # ----------------------------------------------------
        # Wait for AUDIO_START
        # ----------------------------------------------------

        audio_started = False

        while not stop_event.is_set():

            line = read_line(
                timeout_seconds=5
            )

            if line is None:
                break

            print(
                "ESP32:",
                line
            )

            if "AUDIO_START" in line:

                audio_started = True
                break

        if stop_event.is_set():
            break

        if not audio_started:

            print(
                "AUDIO_START not received."
            )

            continue

        # ----------------------------------------------------
        # Receive exactly 96,000 bytes
        # ----------------------------------------------------

        print(
            "Receiving audio:",
            AUDIO_BYTES,
            "bytes"
        )

        current_state = "PROCESSING"

        audio_bytes = read_exact_audio(
            AUDIO_BYTES
        )

        if stop_event.is_set():
            break

        if audio_bytes is None:

            print(
                "Audio reception failed."
            )

            continue

        print(
            "Audio received:",
            len(audio_bytes),
            "bytes"
        )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            process_audio(
                audio_bytes
            )

        except Exception as e:

            print()
            print(
                "INFERENCE ERROR:",
                e
            )

            with state_lock:
                current_state = "ERROR"

        # ----------------------------------------------------
        # Read AUDIO_END / READY
        # ----------------------------------------------------

        for _ in range(3):

            if stop_event.is_set():
                break

            line = read_line(
                timeout_seconds=1
            )

            if line is None:
                break

            print(
                "ESP32:",
                line
            )

            if "READY" in line:
                break

        if not stop_event.is_set():

            current_state = "LISTENING"

            # Small delay prevents serial flooding
            time.sleep(0.05)

    # ========================================================
    # WORKER STOP
    # ========================================================

    print()
    print("TRONIX detection worker stopped.")

    safe_close_serial()

    current_state = "STANDBY"
    system_running = False

    worker_thread = None


# ============================================================
# START SYSTEM
# ============================================================

@app.route("/api/start", methods=["POST"])
def start_system():

    global worker_thread
    global system_running
    global current_state
    global stop_event
    global last_error

    # --------------------------------------------------------
    # Already running
    # --------------------------------------------------------

    if system_running:

        return jsonify({
            "success": True,
            "running": True,
            "message": "TRONIX is already running."
        })

    # --------------------------------------------------------
    # Clean old worker
    # --------------------------------------------------------

    if worker_thread is not None:

        if worker_thread.is_alive():

            return jsonify({
                "success": True,
                "running": True,
                "message": "TRONIX is already starting."
            })

        worker_thread = None

    # --------------------------------------------------------
    # New stop event
    # --------------------------------------------------------

    stop_event = threading.Event()

    system_running = True
    current_state = "CONNECTING"
    last_error = ""

    # --------------------------------------------------------
    # Start exactly ONE worker
    # --------------------------------------------------------

    worker_thread = threading.Thread(
        target=detection_worker,
        daemon=True,
        name="TRONIX-Detection"
    )

    worker_thread.start()

    print(
        "TRONIX detection worker started."
    )

    return jsonify({
        "success": True,
        "running": True,
        "message": "TRONIX starting."
    })


# ============================================================
# STOP SYSTEM
# ============================================================

@app.route("/api/stop", methods=["POST"])
def stop_system():

    global system_running
    global current_state
    global worker_thread

    print()
    print("STOP requested from website.")

    # --------------------------------------------------------
    # Signal worker
    # --------------------------------------------------------

    stop_event.set()

    system_running = False
    current_state = "STOPPING"

    # --------------------------------------------------------
    # Close serial immediately.
    # This helps interrupt blocking serial operations.
    # --------------------------------------------------------

    safe_close_serial()

    # --------------------------------------------------------
    # Do not wait for worker indefinitely.
    # --------------------------------------------------------

    if worker_thread is not None:

        if worker_thread.is_alive():

            worker_thread.join(
                timeout=1.5
            )

    worker_thread = None

    system_running = False
    current_state = "STANDBY"

    print(
        "TRONIX stopped."
    )

    return jsonify({
        "success": True,
        "running": False,
        "message": "TRONIX stopped."
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/api/status", methods=["GET"])
def status():

    with state_lock:

        running = system_running

        return jsonify({

            "running": running,

            "connected": esp32_connected,

            "state": current_state,

            "prediction": last_prediction,

            "prediction_display": DISPLAY_LABELS.get(
                last_prediction,
                last_prediction
            ),

            "confidence": round(
                last_confidence * 100,
                2
            ),

            "rms": round(
                last_rms,
                2
            ),

            "latency_ms": round(
                last_latency_ms,
                2
            ),

            "probabilities": {
                "C0": round(
                    last_probabilities["C0"] * 100,
                    2
                ),
                "C1": round(
                    last_probabilities["C1"] * 100,
                    2
                ),
                "C2": round(
                    last_probabilities["C2"] * 100,
                    2
                ),
                "C3": round(
                    last_probabilities["C3"] * 100,
                    2
                )
            },

            "wake_count": wake_count,

            "total_cycles": total_cycles,

            "last_detection": last_detection_time,

            "error": last_error,

            "performance": {
                "cpu_percent": last_performance["cpu_percent"],
                "free_heap": last_performance["free_heap"],
                "min_free_heap": last_performance["min_free_heap"],
                "largest_free_block": last_performance["largest_free_block"],
                "total_heap": last_performance["total_heap"],
                "source": last_performance["source"],
                "timestamp": last_performance["timestamp"],
            },

            "model": {
                "name": "TRONIX 94-frame INT8",
                "size_kb": 34.39,
                "input": "13 × 94",
                "sample_rate": "16 kHz",
                "classes": 4
            }
        })


# ============================================================
# EVENT LOGS
# ============================================================

@app.route("/api/logs", methods=["GET"])
def get_logs():

    with state_lock:

        return jsonify({
            "logs": event_logs
        })


# ============================================================
# PERSISTENT EVENT LOGS
# ============================================================

@app.route("/api/events", methods=["GET"])
def get_events():
    try:
        if not os.path.exists(EVENTS_FILE):
            return jsonify([])

        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            events = json.load(f)

        return jsonify(events)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# CLEAR LOGS
# ============================================================

@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():

    global event_logs

    with state_lock:
        event_logs = []

    try:
        os.makedirs("data", exist_ok=True)
        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    except Exception as e:
        print("EVENT FILE CLEAR ERROR:", e)
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

    return jsonify({
        "success": True
    })


# ============================================================
# PERFORMANCE API
# ============================================================

@app.route("/api/performance", methods=["GET"])
def performance_status():
    with state_lock:
        performance = dict(last_performance)

    age = (
        time.time() - last_performance_update
        if last_performance_update > 0
        else None
    )

    # A value is called LIVE only when a fresh PERF packet was
    # actually received recently. Otherwise the website clearly
    # presents the last verified measurement.
    live = (
        age is not None
        and age <= PERFORMANCE_LIVE_TIMEOUT
        and esp32_connected
    )

    if live:
        performance["source"] = "LIVE ESP32"
    else:
        performance["source"] = "LAST VERIFIED"

    performance["live"] = live
    performance["connected"] = esp32_connected
    performance["age_seconds"] = (
        round(age, 1) if age is not None else None
    )

    return jsonify(performance)


# ============================================================
# MODEL STATUS
# ============================================================

@app.route("/api/model-status", methods=["GET"])
def model_status():

    return jsonify({

        "loaded": interpreter is not None,

        "model_path": MODEL_PATH,

        "model_exists": os.path.exists(
            MODEL_PATH
        ),

        "model_size_kb": (
            round(
                os.path.getsize(
                    MODEL_PATH
                ) / 1024,
                2
            )
            if os.path.exists(MODEL_PATH)
            else 0
        ),

        "input_shape": (
            input_details[0]["shape"].tolist()
            if input_details is not None
            else None
        ),

        "output_shape": (
            output_details[0]["shape"].tolist()
            if output_details is not None
            else None
        ),

        "features": "13 × 94 MFCC",

        "sample_rate": 16000,

        "quantization": "INT8",

        "keyword": "Hey Leo"
    })


# ============================================================
# TEST AUDIO
# ============================================================

@app.route("/api/test", methods=["POST"])
def test_audio():

    global stop_event

    # --------------------------------------------------------
    # Don't start a second system.
    # --------------------------------------------------------

    if system_running:

        return jsonify({
            "success": False,
            "message": "Stop the live system before testing."
        }), 400

    try:

        if not connect_esp32():

            return jsonify({
                "success": False,
                "message": "ESP32 connection failed."
            }), 500

        # Handshake
        send_command("H")

        ready = wait_for_ready()

        if not ready:

            safe_close_serial()

            return jsonify({
                "success": False,
                "message": "ESP32 did not become ready."
            }), 500

        # Record
        send_command("R")

        # Wait recording start
        while True:

            line = read_line(
                timeout_seconds=5
            )

            if line is None:
                break

            print(
                "ESP32:",
                line
            )

            if "RECORDING_START" in line:
                break

        # Wait recording done
        while True:

            line = read_line(
                timeout_seconds=5
            )

            if line is None:
                break

            print(
                "ESP32:",
                line
            )

            if "RECORDING_DONE" in line:
                break

        # Wait audio start
        while True:

            line = read_line(
                timeout_seconds=5
            )

            if line is None:
                break

            print(
                "ESP32:",
                line
            )

            if "AUDIO_START" in line:
                break

        # Audio
        audio_bytes = read_exact_audio(
            AUDIO_BYTES
        )

        if audio_bytes is None:

            safe_close_serial()

            return jsonify({
                "success": False,
                "message": "Audio reception failed."
            }), 500

        result = process_audio(
            audio_bytes
        )

        # Clean up
        safe_close_serial()

        return jsonify({

            "success": True,

            "prediction": last_prediction,

            "prediction_display": DISPLAY_LABELS.get(
                last_prediction,
                last_prediction
            ),

            "confidence": round(
                last_confidence * 100,
                2
            ),

            "rms": round(
                last_rms,
                2
            ),

            "latency_ms": round(
                last_latency_ms,
                2
            ),

            "probabilities": {
                key: round(
                    value * 100,
                    2
                )
                for key, value
                in last_probabilities.items()
            }
        })

    except Exception as e:

        print(
            "TEST ERROR:",
            e
        )

        safe_close_serial()

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "application": "TRONIX",
        "model_loaded": interpreter is not None
    })


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("              TRONIX - EDGE AI SECURITY")
    print("=" * 60)
    print()
    print("Model:")
    print(MODEL_PATH)
    print()
    print("ESP32:")
    print(PORT)
    print(BAUD)
    print()
    print("Website:")
    print("http://127.0.0.1:5000")
    print()
    print("Performance API:")
    print("http://127.0.0.1:5000/api/performance")
    print()
    print("Press CTRL+C to stop the server.")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )