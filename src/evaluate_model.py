from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = load_model("models/jump_classifier.h5")  # Load the trained model
    X_test = np.load("data/test_data.npy")  # Update to your test data
    y_test = ...  # Load your test labels

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Example plotting
    # Assuming you have accuracy data over epochs
    epochs = np.arange(1, 31)  # Change to your actual number of epochs
    training_accuracy = [...]  # Fill with your training accuracy data
    validation_accuracy = [...]  # Fill with your validation accuracy data

    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
