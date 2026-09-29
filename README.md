# Wireless Signal ML Detector

A small-scale, hardware-free machine-learning project for wireless signal detection and classification.

## Project idea

The application generates controlled synthetic signal samples, adds Gaussian noise at a selected SNR, extracts signal-processing features, and uses a Random Forest classifier to identify the signal class.

Classes:

- NOISE
- AM
- FM
- BPSK
- FSK

The application records every detection in SQLite.

## Architecture

Browser -> Flask API -> Signal Generator -> Feature Extraction -> Random Forest -> Prediction -> Browser

## Requirements

Python 3.10 or newer is recommended.

## Installation

Create a virtual environment:

    python -m venv venv

Windows:

    venv\Scripts\activate

Linux/macOS:

    source venv/bin/activate

Install packages:

    pip install -r requirements.txt

Run:

    python app.py

Open:

    http://127.0.0.1:5000

## Important project limitation

This prototype does not physically sense radio-frequency energy. It simulates and classifies signal data in software. A future version can replace generate_signal() with real IQ samples obtained from an SDR.

## Suggested academic project title

Design and Implementation of a Web-Based Machine-Learning System for Wireless Signal Detection and Classification Using Simulated IQ Data
