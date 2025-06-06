from pathlib import Path
import tensorflow as tf


MODEL_PATH = Path(__file__).parent / "model" / "model.tflite"


def try_load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"❌ Ошибка: {e}")

    try:
        with open(MODEL_PATH, "rb") as f:
            model_content = f.read()
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise


interpreter = try_load_model()
