import os
import numpy as np
import cv2
import tensorflow as tf

MODEL_H5   = "smart_tap_model.h5"
TFLITE_OUT = "smart_tap_int8.tflite"
CLASSES    = ["empty", "half", "full"]
IMG_SIZE   = 224



def convert_to_tflite(data_dir: str = "dataset"):
    """
    Converts .h5 → INT8 quantized TFLite model.
    INT8 quantization reduces model size ~4x and speeds up inference
    significantly on Raspberry Pi (no GPU needed).
    """
    model = tf.keras.models.load_model(MODEL_H5)

    # Representative dataset for INT8 calibration
    def representative_dataset():
        for class_name in CLASSES:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            files = os.listdir(class_dir)[:30]
            for fname in files:
                img = cv2.imread(os.path.join(class_dir, fname))
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                yield [np.expand_dims(img, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_model)

    original_mb = os.path.getsize(MODEL_H5) / 1e6
    tflite_mb   = os.path.getsize(TFLITE_OUT) / 1e6
    print(f"Original:  {original_mb:.1f} MB")
    print(f"TFLite INT8: {tflite_mb:.1f} MB  ({original_mb/tflite_mb:.1f}x smaller)")
    print(f"Saved to {TFLITE_OUT}")



class TFLitePredictor:
    def __init__(self, model_path: str = TFLITE_OUT):
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Check if INT8 quantized
        self.is_quantized = self.input_details[0]["dtype"] == np.uint8
        print(f"Model loaded. Quantized: {self.is_quantized}")

    def predict(self, frame: np.ndarray) -> dict:
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_quantized:
            # INT8: scale to 0-255 uint8
            img = img.astype(np.uint8)
        else:
            img = (img.astype(np.float32) / 255.0)

        img = np.expand_dims(img, 0)
        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        if self.is_quantized:
            # Dequantize output
            scale, zero_point = self.output_details[0]["quantization"]
            output = scale * (output.astype(np.float32) - zero_point)

        idx = int(np.argmax(output))
        return {
            "label":      CLASSES[idx],
            "confidence": round(float(output[idx]), 3),
            "all_probs":  {c: round(float(p), 3) for c, p in zip(CLASSES, output)},
        }



def run_pi_live(camera_index: int = 0):
    """
    Optimized live loop for Raspberry Pi.
    Runs TFLite every 10 frames (~3 predictions/sec at 30fps).
    Uses classical CV on remaining frames for zero-latency feedback.
    """
    from smart_tap import TagDetector, preprocess_image

    predictor = TFLitePredictor()
    detector  = TagDetector()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)   # lower res for Pi
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)

    frame_count = 0
    result = {"label": "...", "confidence": 0}
    COLORS = {"empty": (0,200,50), "half": (0,180,255), "full": (50,50,255)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            # TFLite inference
            result = predictor.predict(frame)
        elif frame_count % 2 == 0:
            # Classical CV on off-frames (nearly free)
            cv_out = detector.detect(frame)
            if cv_out["confidence"] > 0.8:
                result = {"label": cv_out["label"],
                          "confidence": cv_out["confidence"]}

        frame_count += 1

        color = COLORS.get(result["label"], (200,200,200))
        label = result["label"].upper()
        conf  = result["confidence"]

        cv2.putText(frame, f"{label} ({conf:.0%})",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imshow("Smart Tap Pi", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        convert_to_tflite()
    else:
        run_pi_live()
