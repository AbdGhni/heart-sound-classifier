# Heart sound classification (log-mel CNN)

Binary classifier for short phonocardiogram recordings (normal vs abnormal), using a small CNN on log-mel spectrograms, with a web demo.

> **Research prototype. Not a medical device. Not for clinical use.** Trained on a small public dataset only; no patient data is included in this repository.

**Live demo:** _Hugging Face Space link to add_

## Results (5-fold stratified cross-validation, 545 recordings)

Out-of-fold predictions at threshold 0.5, 95% confidence intervals from 1,000 bootstrap resamples.

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 65.7% | 61.7 to 69.7% |
| Sensitivity (abnormal) | 68.0% | 61.2 to 74.2% |
| Specificity | 64.4% | 59.0 to 69.2% |
| AUC-ROC | 0.743 | 0.698 to 0.787 |

Confusion matrix: TN 226, FP 125, FN 62, TP 132 (`results/metrics.json`, figures in `results/`).

These are modest results. This is a baseline, and the repository is meant to show the full pipeline and an honest evaluation rather than a strong classifier.

## Method

- Data: Kaggle `kinguistics/heartbeat-sounds` (PASCAL Classifying Heart Sounds Challenge, sets A and B). 351 normal and 194 abnormal (129 murmur, 46 extrasystole, 19 extra heart sound) recordings. The 40 noisy "artifact" recordings are excluded and the unlabelled test files are not used.
- Features: audio resampled to 2 kHz, first 5 s (padded), peak-normalised, 64-band log-mel spectrogram.
- Model: 3 convolutional blocks (16, 32, 64 filters) with batch norm, global average pooling, dropout 0.4, sigmoid output; class-weighted loss; 40 epochs. The demo model is retrained on all recordings with the same recipe.
- Evaluation: 5-fold stratified cross-validation.

## Limitations (read before quoting the numbers)

- A few hundred recordings, with wide confidence intervals.
- No patient identifiers, so recordings from the same patient may be in different folds, which can inflate the scores.
- Recordings come from two sources with different devices and noise levels (a digital stethoscope and a phone app); the model may partly learn the source.
- Normal vs abnormal only: no murmur grading and no diagnosis. A published model would need segmentation of heart cycles and an external test set.

## Run it

Train (Google Colab): open `heart_sound_training.ipynb` and run all cells. It saves `heart_sound_cnn.keras` and `metrics.json`.

Demo locally:

```bash
cd demo
pip install -r requirements.txt
# put heart_sound_cnn.keras in this folder
python app.py
```

## Author

Abdullah Ghani, data and AI engineering student at ECE Paris.
