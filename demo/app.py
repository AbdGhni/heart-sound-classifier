"""Gradio demo: normal vs abnormal heart sound recording (research prototype, not for clinical use)."""
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr

SR, DUR, N_MELS = 2000, 5, 64
MODEL = tf.keras.models.load_model("heart_sound_cnn.keras")
DISCLAIMER = ("Research prototype trained on a few hundred public recordings. "
              "Not a medical device. Not for clinical use.")


def features(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=DUR * 2)
    y = y[: SR * DUR]
    y = np.pad(y, (0, max(0, SR * DUR - len(y))))
    y = y / (np.abs(y).max() + 1e-8)
    m = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=256, hop_length=32, n_mels=N_MELS, fmax=SR // 2), ref=np.max)
    return ((m + 80) / 80).astype("float32")


def predict(path):
    if path is None:
        return {}, None
    f = features(path)
    p = float(MODEL.predict(f[None, ..., None], verbose=0)[0, 0])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.imshow(f, origin="lower", aspect="auto")
    ax.set_title("Log-mel spectrogram (first 5 s)")
    ax.set_xlabel("time"); ax.set_ylabel("mel band")
    fig.tight_layout()
    return {"Abnormal": p, "Normal": 1 - p}, fig


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Heart sound recording (.wav)"),
    outputs=[gr.Label(label="Model output"), gr.Plot(label="Input spectrogram")],
    title="Heart sound classification (log-mel CNN)",
    description=DISCLAIMER,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
