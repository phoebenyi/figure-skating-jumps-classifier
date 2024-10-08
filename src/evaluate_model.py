from tensorflow.keras.models import load_model
import numpy as np

if __name__ == "__main__":
    model = load_model("models/jump_classifier.h5")
    X_test = np.load("data/test_data.npy")
    y_test = ...

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
