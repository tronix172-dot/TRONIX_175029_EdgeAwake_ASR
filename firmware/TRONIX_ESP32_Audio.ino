#include <Arduino.h>
#include "driver/i2s.h"

// ============================================================
// TRONIX ESP32 + INMP441 AUDIO STREAMER
// ============================================================

// ---------------- I2S ----------------

#define I2S_PORT I2S_NUM_0

#define I2S_BCLK 26
#define I2S_WS   25
#define I2S_SD   33

#define SAMPLE_RATE 16000

// ---------------- AUDIO ----------------

#define RECORD_SECONDS 3
#define AUDIO_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define AUDIO_BYTES (AUDIO_SAMPLES * sizeof(int16_t))

// ---------------- STATE ----------------

bool streaming = false;

// ============================================================
// I2S SETUP
// ============================================================

void setupI2S()
{
    i2s_config_t config =
    {
        .mode =
            (i2s_mode_t)(
                I2S_MODE_MASTER |
                I2S_MODE_RX
            ),

        .sample_rate = SAMPLE_RATE,

        .bits_per_sample =
            I2S_BITS_PER_SAMPLE_32BIT,

        .channel_format =
            I2S_CHANNEL_FMT_ONLY_LEFT,

        .communication_format =
            I2S_COMM_FORMAT_I2S,

        .intr_alloc_flags =
            ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = 8,

        .dma_buf_len = 512,

        .use_apll = false,

        .tx_desc_auto_clear = false,

        .fixed_mclk = 0
    };

    i2s_pin_config_t pins =
    {
        .bck_io_num = I2S_BCLK,

        .ws_io_num = I2S_WS,

        .data_out_num =
            I2S_PIN_NO_CHANGE,

        .data_in_num = I2S_SD
    };

    i2s_driver_install(
        I2S_PORT,
        &config,
        0,
        NULL
    );

    i2s_set_pin(
        I2S_PORT,
        &pins
    );

    i2s_zero_dma_buffer(
        I2S_PORT
    );
}

// ============================================================
// READ ONE SAMPLE
// ============================================================

int16_t readSample()
{
    int32_t raw = 0;

    size_t bytesRead = 0;

    esp_err_t result =
        i2s_read(
            I2S_PORT,
            &raw,
            sizeof(raw),
            &bytesRead,
            portMAX_DELAY
        );

    if (
        result != ESP_OK ||
        bytesRead != sizeof(raw)
    )
    {
        return 0;
    }

    // INMP441 gives useful data in upper bits
    int32_t sample = raw >> 14;

    if (sample > 32767)
        sample = 32767;

    if (sample < -32768)
        sample = -32768;

    return (int16_t)sample;
}

// ============================================================
// SEND ONE 3-SECOND RECORDING
// ============================================================

void recordAndSend()
{
    Serial.println("RECORDING_START");

    delay(20);

    int16_t *audio =
        (int16_t *)malloc(
            AUDIO_BYTES
        );

    if (audio == nullptr)
    {
        Serial.println(
            "ERROR: AUDIO BUFFER ALLOCATION FAILED"
        );

        return;
    }

    // -------------------------------
    // Capture exactly 3 seconds
    // -------------------------------

    for (
        int i = 0;
        i < AUDIO_SAMPLES;
        i++
    )
    {
        audio[i] = readSample();
    }

    Serial.println("RECORDING_DONE");

    delay(20);

    Serial.println("AUDIO_START");

    delay(10);

    // -------------------------------
    // Send raw PCM16
    // -------------------------------

    Serial.write(
        (uint8_t *)audio,
        AUDIO_BYTES
    );

    Serial.flush();

    Serial.println();

    Serial.println("AUDIO_END");

    delay(20);

    Serial.println("READY");

    free(audio);
}

// ============================================================
// CONTINUOUS STREAM
// ============================================================

void startContinuousStream()
{
    if (streaming)
        return;

    streaming = true;

    Serial.println("STREAM_START");

    delay(20);

    while (streaming)
    {
        // Check for STOP command.
        if (Serial.available())
        {
            char command =
                Serial.read();

            if (
                command == 'S' ||
                command == 's'
            )
            {
                streaming = false;

                Serial.println();

                Serial.println(
                    "STREAM_STOP"
                );

                break;
            }
        }

        // ---------------------------
        // Read a small block
        // ---------------------------

        int16_t samples[256];

        for (int i = 0; i < 256; i++)
        {
            samples[i] =
                readSample();

            // Check STOP periodically
            if (Serial.available())
            {
                char command =
                    Serial.read();

                if (
                    command == 'S' ||
                    command == 's'
                )
                {
                    streaming = false;
                    break;
                }
            }
        }

        if (!streaming)
            break;

        // ---------------------------
        // Send binary PCM16
        // ---------------------------

        Serial.write(
            (uint8_t *)samples,
            sizeof(samples)
        );
    }
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(921600);

    delay(1500);

    setupI2S();

    Serial.println();
    Serial.println(
        "================================"
    );

    Serial.println(
        "TRONIX ESP32 AUDIO SYSTEM"
    );

    Serial.println(
        "INMP441 READY"
    );

    Serial.println(
        "SAMPLE RATE: 16000 Hz"
    );

    Serial.println(
        "================================"
    );
}

// ============================================================
// LOOP
// ============================================================

void loop()
{
    if (Serial.available())
    {
        char command =
            Serial.read();

        // ---------------------------
        // HANDSHAKE
        // ---------------------------

        if (
            command == 'H' ||
            command == 'h'
        )
        {
            Serial.println(
                "ESP32_READY"
            );
        }

        // ---------------------------
        // CONTINUOUS STREAM
        // ---------------------------

        else if (
            command == 'C' ||
            command == 'c'
        )
        {
            startContinuousStream();
        }

        // ---------------------------
        // STOP
        // ---------------------------

        else if (
            command == 'S' ||
            command == 's'
        )
        {
            streaming = false;

            Serial.println(
                "STREAM_STOP"
            );
        }

        // ---------------------------
        // SINGLE RECORDING
        // ---------------------------

        else if (
            command == 'R' ||
            command == 'r'
        )
        {
            recordAndSend();
        }
    }

    delay(2);
}