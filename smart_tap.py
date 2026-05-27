

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 20
CLASSES     = ["empty", "half", "full"]   # folder names in your dataset
DATA_DIR    = "dataset"                   # root folder containing class subfolders
MODEL_PATH  = "smart_tap_model.h5"

# HSV ranges for rainbow tag colors (tune with color_tuner.py if needed)
TAG_COLORS_HSV = {
    "red1":   ([0,  120, 70],  [10, 255, 255]),
    "red2":   ([170,120, 70],  [180,255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255]),
    "blue":   ([100, 80,  50], [130,255, 255]),
}


#
class TagDetector:
    """
    Detects rainbow tag pixels using HSV masking.
    Returns tag pixel count → used to infer water level.
    """

    def detect(self, frame: np.ndarray) -> dict:
        """
        Args:
            frame: BGR image (from cv2.imread or camera)
        Returns:
            dict with pixel_count, ratio, label, confidence
        """
        # Crop to center region (tag is always in center of bucket)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        roi_size = min(h, w) // 3
        roi = frame[cy - roi_size:cy + roi_size, cx - roi_size:cx + roi_size]

        # Apply CLAHE for lighting robustness
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        # Combine all tag color masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for name, (lo, hi) in TAG_COLORS_HSV.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        total_pixels = roi.shape[0] * roi.shape[1]
        tag_pixels   = int(np.sum(combined_mask > 0))
        ratio        = tag_pixels / total_pixels

        # Decision thresholds (tune based on your dataset)
        if ratio > 0.05:
            label, confidence = "empty", min(ratio / 0.15, 1.0)
        elif ratio > 0.015:
            label, confidence = "half",  0.6
        else:
            label, confidence = "full",  min((0.015 - ratio) / 0.015, 1.0)

        return {
            "label":       label,
            "confidence":  round(confidence, 3),
            "tag_pixels":  tag_pixels,
            "ratio":       round(ratio, 4),
            "mask":        combined_mask,
        }



def build_model(num_classes: int = 3) -> tf.keras.Model:
    """
    MobileNetV2 with custom head. Pretrained on ImageNet.
    Only 3 output classes: empty, half, full.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    # Phase 1: freeze entire base
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base


def fine_tune_model(model, base, unfreeze_from: int = -30):
    """Unfreeze top layers of base for fine-tuning."""
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # much lower LR
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Resize, CLAHE, normalize. Applied to every image."""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def get_data_generators(data_dir: str):
    """Creates train/val generators with augmentation."""
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        preprocessing_function=lambda x: preprocess_image(x.astype(np.uint8)),
        rotation_range=10,
        brightness_range=[0.7, 1.3],
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        classes=CLASSES
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        classes=CLASSES
    )

    return train_gen, val_gen



def train(data_dir: str = DATA_DIR):
    train_gen, val_gen = get_data_generators(data_dir)

    model, base = build_model(num_classes=len(CLASSES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7
        ),
    ]

    print("\n── Phase 1: Training head only ──")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

    print("\n── Phase 2: Fine-tuning top layers ──")
    model = fine_tune_model(model, base, unfreeze_from=-30)
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=10,
        callbacks=callbacks
    )

    plot_history(history1, history2)
    return model



def evaluate(model, data_dir: str = DATA_DIR):
    _, val_gen = get_data_generators(data_dir)
    val_gen.reset()

    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    print("\n── Classification Report ──")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")


class SmartTapPredictor:
    """
    Hybrid predictor:
    - Classical CV runs first (fast)
    - If confidence < threshold, falls back to CNN
    """

    def __init__(self, model_path: str = MODEL_PATH, cv_threshold: float = 0.75):
        self.detector    = TagDetector()
        self.cv_threshold = cv_threshold
        self.model       = None
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded CNN model from {model_path}")
        else:
            print("No CNN model found — running classical CV only")

    def predict(self, frame: np.ndarray) -> dict:
        cv_result = self.detector.detect(frame)

        if cv_result["confidence"] >= self.cv_threshold:
            return {
                "label":    cv_result["label"],
                "confidence": cv_result["confidence"],
                "method":   "classical_cv",
                "cv_ratio": cv_result["ratio"],
            }

        # Fall back to CNN
        if self.model is not None:
            img = preprocess_image(frame)
            img = np.expand_dims(img, 0)
            probs = self.model.predict(img, verbose=0)[0]
            idx   = int(np.argmax(probs))
            return {
                "label":      CLASSES[idx],
                "confidence": round(float(probs[idx]), 3),
                "method":     "cnn",
                "cv_ratio":   cv_result["ratio"],
                "all_probs":  {c: round(float(p), 3) for c, p in zip(CLASSES, probs)},
            }

        # No model: return CV result anyway
        return {**cv_result, "method": "classical_cv_low_conf"}



def run_live(camera_index: int = 0):
    """Real-time inference from webcam / USB camera."""
    predictor = SmartTapPredictor()
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    STATUS_COLOR = {"empty": (0, 200, 50), "half": (0, 180, 255), "full": (50, 50, 255)}
    LABEL_TEXT   = {"empty": "EMPTY — Tap ON",
                    "half":  "HALF  — Tap ON",
                    "full":  "FULL  — Tap OFF"}
    frame_count  = 0
    result       = {"label": "...", "confidence": 0, "method": "—"}

    print("Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict every 5 frames for speed
        if frame_count % 5 == 0:
            result = predictor.predict(frame)
        frame_count += 1

        # Overlay
        label  = result["label"]
        color  = STATUS_COLOR.get(label, (200, 200, 200))
        text   = LABEL_TEXT.get(label, label)
        conf   = result["confidence"]
        method = result.get("method", "")

        cv2.rectangle(frame, (0, 0), (640, 60), (20, 20, 20), -1)
        cv2.putText(frame, text,   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{conf:.0%}  [{method}]",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Draw ROI box
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        rs = min(h, w) // 3
        cv2.rectangle(frame, (cx - rs, cy - rs), (cx + rs, cy + rs), color, 1)

        cv2.imshow("Smart Tap", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def plot_history(*histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for h in histories:
        ax1.plot(h.history["accuracy"],     label="train acc")
        ax1.plot(h.history["val_accuracy"], label="val acc")
        ax2.plot(h.history["loss"],         label="train loss")
        ax2.plot(h.history["val_loss"],     label="val loss")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.set_title("Loss");     ax2.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved training_curves.png")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smart Tap")
    parser.add_argument("mode", choices=["train", "evaluate", "live", "predict"])
    parser.add_argument("--image", help="Path to image for predict mode")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "train":
        model = train()
        evaluate(model)

    elif args.mode == "evaluate":
        model = tf.keras.models.load_model(MODEL_PATH)
        evaluate(model)

    elif args.mode == "live":
        run_live(args.camera)

    elif args.mode == "predict":
        if not args.image:
            print("Provide --image path")
        else:
            frame = cv2.imread(args.image)
            predictor = SmartTapPredictor()
            result = predictor.predict(frame)
            print(result)
