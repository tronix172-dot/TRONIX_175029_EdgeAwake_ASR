import serial
import time


# =========================================================
# TRONIX ESP32 SERIAL TEST
# =========================================================

PORT = "COM9"
BAUD = 921600


print()
print("=" * 55)
print("TRONIX ESP32 CONNECTION TEST")
print("=" * 55)
print(f"Port : {PORT}")
print(f"Baud : {BAUD}")
print("=" * 55)


try:

    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=2
    )

    print()
    print("COM9 opened successfully.")
    print("Waiting for ESP32...")
    print()


    # Give ESP32 a moment

    time.sleep(2)


    # Clear old serial data

    ser.reset_input_buffer()


    # =====================================================
    # SEND HANDSHAKE
    # =====================================================

    print("Sending: H")

    ser.write(b"H")
    ser.flush()


    # =====================================================
    # WAIT FOR ESP32 RESPONSE
    # =====================================================

    start_time = time.time()

    received = False


    while time.time() - start_time < 5:

        if ser.in_waiting:

            line = ser.readline()

            try:
                text = line.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

            except Exception:
                text = str(line)


            if text:

                print(
                    f"ESP32 → {text}"
                )


                if "ESP32_READY" in text:

                    received = True

                    break


    # =====================================================
    # RESULT
    # =====================================================

    print()

    if received:

        print("=" * 55)
        print("SUCCESS")
        print("=" * 55)
        print("Python ↔ ESP32 communication is working.")
        print("COM9 is ready for TRONIX.")
        print("=" * 55)

    else:

        print("=" * 55)
        print("NO HANDSHAKE RECEIVED")
        print("=" * 55)
        print("Python opened COM9, but ESP32_READY")
        print("was not received.")
        print("=" * 55)


    ser.close()

    print()
    print("COM9 closed.")


except serial.SerialException as e:

    print()
    print("=" * 55)
    print("SERIAL ERROR")
    print("=" * 55)
    print(e)
    print("=" * 55)

except Exception as e:

    print()
    print("=" * 55)
    print("ERROR")
    print("=" * 55)
    print(e)
    print("=" * 55)