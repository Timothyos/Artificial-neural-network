import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="Fashion-MNIST Lab", page_icon="👕", layout="wide")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
EXPERIMENTS = [
    "Baseline",
    "Dropout",
    "Early stopping",
    "Batch normalization",
    "Data augmentation",
]

st.title("👕 Fashion-MNIST Neural Network Lab")
st.caption("An interactive companion to the original regularization and model-training experiments.")

@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # Keep the official test set untouched. Create validation data only from training data.
    x_fit, x_val = x_train[:-6000], x_train[-6000:]
    y_fit, y_val = y_train[:-6000], y_train[-6000:]
    return x_fit, y_fit, x_val, y_val, x_test, y_test

def build_model(experiment):
    model = keras.Sequential(name=experiment.lower().replace(" ", "_"))
    model.add(layers.Input(shape=(28, 28)))
    if experiment == "Data augmentation":
        model.add(layers.RandomRotation(0.10))
        model.add(layers.RandomZoom(0.10))
        model.add(layers.RandomTranslation(0.10, 0.10))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
    elif experiment == "Batch normalization":
        model.add(layers.Flatten())
        model.add(layers.Dense(256, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dense(128, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
    elif experiment == "Dropout":
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.3))
    else:
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

@st.cache_resource(show_spinner=False)
def train_experiment(experiment, epochs, seed):
    keras.utils.set_random_seed(seed)
    x_fit, y_fit, x_val, y_val, x_test, y_test = load_data()
    model = build_model(experiment)
    callbacks = []
    if experiment == "Early stopping":
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ))
    if experiment == "Data augmentation":
        history = model.fit(
            x_fit, y_fit, validation_data=(x_val, y_val),
            epochs=epochs, batch_size=128, verbose=0, callbacks=callbacks
        )
    else:
        history = model.fit(
            x_fit, y_fit, validation_data=(x_val, y_val),
            epochs=epochs, batch_size=128, verbose=0, callbacks=callbacks
        )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return model, history.history, float(test_loss), float(test_accuracy)

with st.sidebar:
    st.header("Experiment settings")
    experiment = st.selectbox("Training strategy", EXPERIMENTS)
    epochs = st.slider("Maximum epochs", min_value=3, max_value=20, value=8)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
    st.caption("Training runs when you click the button. Results are cached for the selected settings.")
    run = st.button("Train model", type="primary", use_container_width=True)

tab_train, tab_predict, tab_about = st.tabs(["📊 Train & analyse", "🔎 Try an image", "ℹ️ About"])

if "trained_key" not in st.session_state:
    st.session_state.trained_key = None

with tab_train:
    st.subheader("Compare regularization strategies")
    st.write("Train a Fashion-MNIST classifier and evaluate it on the official test set.")
    if run:
        with st.spinner("Training the neural network. The first run downloads Fashion-MNIST and may take a few minutes..."):
            try:
                model, history, test_loss, test_accuracy = train_experiment(
                    experiment, int(epochs), int(seed)
                )
                st.session_state.trained_key = (experiment, int(epochs), int(seed))
                st.session_state.test_loss = test_loss
                st.session_state.test_accuracy = test_accuracy
                st.success("Training and test evaluation completed.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    key = (experiment, int(epochs), int(seed))
    if st.session_state.trained_key == key:
        model, history, test_loss, test_accuracy = train_experiment(*key)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{test_accuracy * 100:.2f}%")
        c2.metric("Test loss", f"{test_loss:.4f}")
        c3.metric("Epochs completed", len(history["loss"]))
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots()
            ax.plot(history["accuracy"], label="Training")
            ax.plot(history["val_accuracy"], label="Validation")
            ax.set_title("Accuracy by epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fig, ax = plt.subplots()
            ax.plot(history["loss"], label="Training")
            ax.plot(history["val_loss"], label="Validation")
            ax.set_title("Loss by epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
        gap = history["accuracy"][-1] - history["val_accuracy"][-1]
        st.write(f"Final training–validation accuracy gap: **{gap * 100:.2f} percentage points**.")
        st.caption("A large positive gap can indicate overfitting. Compare strategies using the same data split and random seed.")
    else:
        st.info("Choose a strategy in the sidebar, set the maximum epochs, then click **Train model**.")

with tab_predict:
    st.subheader("Classify a 28 × 28 image")
    active_key = st.session_state.trained_key
    if active_key is None:
        st.info("Train a model first in the **Train & analyse** tab.")
    else:
        model, _, _, _ = train_experiment(*active_key)
        upload = st.file_uploader("Upload a clothing image", type=["png", "jpg", "jpeg"])
        if upload is not None:
            image = Image.open(upload).convert("L")
            image = ImageOps.fit(image, (28, 28))
            arr = np.asarray(image).astype("float32") / 255.0
            # Fashion-MNIST uses white clothing on a dark background; invert common dark-on-light uploads.
            if arr.mean() > 0.5:
                arr = 1.0 - arr
            st.image(Image.fromarray((arr * 255).astype("uint8")), caption="28 × 28 model input", width=160)
            probabilities = model.predict(arr[np.newaxis, ...], verbose=0)[0]
            best = int(np.argmax(probabilities))
            st.metric("Predicted class", CLASS_NAMES[best], f"{probabilities[best] * 100:.1f}% confidence")
            prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probabilities})
            prob_df = prob_df.sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Class"))
            st.warning("This model was trained on Fashion-MNIST grayscale images, not general photographs. Predictions on real-world photos may be unreliable.")

with tab_about:
    st.subheader("Project overview")
    st.write(
        "This app explores how regularization and training techniques affect a neural network "
        "trained on the Fashion-MNIST clothing dataset."
    )
    st.markdown("""
**Experiments**
- **Baseline:** two fully connected hidden layers (256 and 128 units).
- **Dropout:** dropout rates of 0.5 and 0.3 to reduce co-adaptation.
- **Early stopping:** monitors validation loss and restores the best weights.
- **Batch normalization:** normalizes hidden-layer activations during training.
- **Data augmentation:** applies small random rotations, zooms, and translations.

**Dataset:** 60,000 training images and 10,000 official test images, each 28 × 28 grayscale pixels across 10 clothing categories.

**Evaluation note:** a validation split is taken from the training set. The official test set is kept separate and used only for final evaluation, avoiding test-set leakage.
""")
    st.markdown("Source project: [Timothyos/Artificial-neural-network](https://github.com/Timothyos/Artificial-neural-network)")
