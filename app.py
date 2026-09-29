from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import numpy as np
from datetime import datetime
from ml_model import SignalClassifier, generate_signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "signals.db")

app = Flask(__name__)
classifier = SignalClassifier()
classifier.train()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_type TEXT NOT NULL,
            predicted_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            snr REAL NOT NULL,
            samples INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(silent=True) or {}
    signal_type = str(data.get("signal_type", "BPSK")).upper()
    snr = float(data.get("snr", 10))
    samples = int(data.get("samples", 2048))

    allowed = ["NOISE", "AM", "FM", "BPSK", "FSK"]
    if signal_type not in allowed:
        return jsonify({"error": "Invalid signal type"}), 400

    samples = max(256, min(samples, 8192))
    snr = max(-10, min(snr, 30))

    signal = generate_signal(signal_type, samples, snr)
    result = classifier.predict(signal)

    conn = get_db()
    conn.execute(
        """INSERT INTO detections
           (signal_type, predicted_type, confidence, snr, samples, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            signal_type,
            result["predicted_type"],
            result["confidence"],
            snr,
            samples,
            datetime.utcnow().isoformat(timespec="seconds")
        )
    )
    conn.commit()
    conn.close()

    # Downsample for browser display.
    display = signal[:min(len(signal), 1000)]
    waveform = np.abs(display).tolist()

    return jsonify({
        "actual_type": signal_type,
        "predicted_type": result["predicted_type"],
        "confidence": result["confidence"],
        "snr": snr,
        "samples": samples,
        "waveform": waveform,
        "features": result["features"],
        "probabilities": result["probabilities"]
    })

@app.route("/api/history")
def history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

if __name__ == "__main__":
    app.run(debug=True)
