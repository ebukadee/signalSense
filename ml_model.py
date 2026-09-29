import numpy as np
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SIGNAL_TYPES = ["NOISE", "AM", "FM", "BPSK", "FSK"]

def add_awgn(signal, snr_db):
    power = np.mean(np.abs(signal) ** 2)
    if power <= 0:
        power = 1.0
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise

def generate_signal(signal_type, n=2048, snr_db=10):
    t = np.arange(n) / 1000.0

    if signal_type == "NOISE":
        return np.sqrt(0.5) * (
            np.random.randn(n) + 1j * np.random.randn(n)
        )

    carrier = 100.0
    symbols = np.random.choice([-1.0, 1.0], size=n)

    if signal_type == "AM":
        mod = 1.0 + 0.7 * np.sin(2 * np.pi * 7 * t)
        signal = mod * np.exp(1j * 2 * np.pi * carrier * t)

    elif signal_type == "FM":
        message = np.sin(2 * np.pi * 7 * t)
        phase = 2 * np.pi * carrier * t + 1.5 * np.cumsum(message) / 1000.0
        signal = np.exp(1j * phase)

    elif signal_type == "BPSK":
        signal = symbols * np.exp(1j * 2 * np.pi * carrier * t)

    elif signal_type == "FSK":
        bits = np.random.choice([0, 1], size=n)
        freq = np.where(bits == 0, carrier - 20, carrier + 20)
        phase = 2 * np.pi * np.cumsum(freq) / 1000.0
        signal = np.exp(1j * phase)

    else:
        raise ValueError("Unknown signal type")

    return add_awgn(signal, snr_db)

def extract_features(x):
    x = np.asarray(x)
    amp = np.abs(x)

    power = float(np.mean(amp ** 2))
    rms = float(np.sqrt(power))
    std = float(np.std(amp))
    peak = float(np.max(amp))
    crest = peak / (rms + 1e-12)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x)))
    spectrum = spectrum + 1e-12
    ps = spectrum ** 2
    ps_norm = ps / np.sum(ps)

    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1/1000.0))
    centroid = float(np.sum(np.abs(freqs) * ps_norm))
    bandwidth = float(
        np.sqrt(np.sum((np.abs(freqs) - centroid) ** 2 * ps_norm))
    )

    entropy = float(
        -np.sum(ps_norm * np.log2(ps_norm))
        / np.log2(len(ps_norm))
    )

    peak_ratio = float(np.max(ps) / np.mean(ps))

    # Phase-change statistics are useful for distinguishing digitally
    # modulated signals from simple AM/FM examples.
    phase = np.unwrap(np.angle(x))
    phase_diff = np.diff(phase)
    phase_std = float(np.std(phase_diff))

    return np.array([
        power, rms, std, peak, crest,
        centroid, bandwidth, entropy,
        peak_ratio, phase_std
    ], dtype=float)

class SignalClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=160,
            max_depth=14,
            random_state=42,
            class_weight="balanced"
        )
        self.accuracy = 0.0

    def train(self):
        X = []
        y = []

        np.random.seed(42)

        for label in SIGNAL_TYPES:
            for _ in range(250):
                n = int(np.random.choice([1024, 2048, 4096]))
                snr = float(np.random.uniform(0, 20))
                sig = generate_signal(label, n, snr)
                X.append(extract_features(sig))
                y.append(label)

        X = np.asarray(X)
        y = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        self.accuracy = float(accuracy_score(y_test, predictions))

    def predict(self, signal):
        features = extract_features(signal)
        prediction = self.model.predict(features.reshape(1, -1))[0]
        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]

        result_probs = {}
        for i, cls in enumerate(self.model.classes_):
            result_probs[str(cls)] = round(float(probabilities[i]) * 100, 2)

        confidence = float(np.max(probabilities) * 100)

        feature_names = [
            "power", "rms", "std_amplitude", "peak", "crest_factor",
            "spectral_centroid", "bandwidth", "spectral_entropy",
            "peak_ratio", "phase_change_std"
        ]

        result_features = {}
        for i, name in enumerate(feature_names):
            result_features[name] = round(float(features[i]), 6)

        return {
            "predicted_type": str(prediction),
            "confidence": round(confidence, 2),
            "probabilities": result_probs,
            "features": result_features
        }
