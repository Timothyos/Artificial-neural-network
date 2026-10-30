# Fashion-MNIST Neural Network Lab — Streamlit app

This deployment adds an interactive Streamlit interface to the original Fashion-MNIST experiments. The original notebook is not modified.

## Run locally

Use Python 3.11 in a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first training run downloads Fashion-MNIST. Choose an experiment in the sidebar and click **Train model**. The app caches trained models for the selected settings during the session.

## Deploy on Streamlit Community Cloud

1. Push this branch to GitHub (or merge it into the default branch).
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select this repository and the branch `streamlit-deployment`.
4. Set the main file path to `app.py`.
5. In advanced settings, choose Python 3.11 if a Python version is requested, then deploy.

## Experiments included

- Baseline dense neural network
- Dropout regularization
- Early stopping
- Batch normalization
- Random image augmentation

The app reserves 6,000 images from the training partition for validation and uses the official 10,000-image test partition only for final evaluation. This avoids using test data as validation data.

## Notes

- TensorFlow training can take a few minutes, especially on the first run.
- Uploaded images are resized to 28 × 28 grayscale. The model is trained on Fashion-MNIST, so predictions on ordinary photographs may be unreliable.
- The app is an educational experiment, not a production image-classification service.
