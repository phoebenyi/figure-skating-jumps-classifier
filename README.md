# figure-skating-jumps-classifier

## Project Overview
Figure Skating Classifier using TensorFlow 


## Project Description:

Differentiating between the six figure skating jumps—Toe Loop, Salchow, Loop, Flip, Lutz, and Axel—can be challenging for newcomers to the sport. These jumps are often integrated into fast-paced routines, making visual differentiation even harder.

As a figure skating and AI/ML enthusiast, I developed a machine learning model utilizing TensorFlow and Keras to classify these jumps using 3D pose data from the FS-Jump3D dataset.

This ML model aims to differentiate between Toe Loop, Salchow, Loop, Flip, Lutz, Axel and Combination jumps.


## Note: (for curious people :p)

- Salchow     -  Backward inside edge

- Loop        -  Backward outside edge

- Flip        -  Backward inside edge, with toe-pick

- Lutz        -  Backward outside edge, with toe-pick, landing on the opposite foot

- Toe Loop    -  Backward outside edge, with toe-pick, landing on the same foot

- Axel        -  Forward outside edge, with an additional half rotation in the air (a single Axel consists of one and a half rotation)


## Dataset Information
FS-Jump3D Dataset provides detailed 3D pose data of skaters performing different jumps and captures key body movements through markerless motion capture technology. This makes it ideal for analyzing complex 3D movements and subtle technical differences between jumps, such as take-off angles, rotation, and landing positions, which are critical for classifying each type of jump accurately.

FS-Jump3D Dataset: https://github.com/ryota-skating/FS-Jump3D


## Technologies Used
- TensorFlow: For building and training the 1D Convolutional Neural Network.

- NumPy: For numerical computing and array handling.

- Scikit-learn: For model evaluation and metrics.

- Matplotlib & Seaborn: For data visualization, including plots and charts.

- Joblib: For saving and loading Python objects, such as models.


## Results
Test Loss: 0.3238, Test Accuracy: 0.8696

Confusion Matrix:

<img width="500" alt="Classification Report" src="https://github.com/user-attachments/assets/ccc5bb99-b583-440c-9e28-4bbc864fff67">

Classification Report:

<img width="500" alt="Classification Report" src="https://github.com/user-attachments/assets/d323973c-5d08-4dac-887a-3500a290803a">



## Installation Instructions
To set up the environment, clone this repository and install the required packages:

```bash
git clone https://github.com/phoebenyi/figure-skating-jumps-classifier.git
cd figure-skating-jumps-classifier
python3 -m pip install -r requirements.txt
```


## Running Instructions:
To preprocess data, train the model, and evaluate it, use the following commands:
```bash
python3 src/preprocess_data.py   # Preprocess data
python3 src/train_model.py        # Train the model
python3 src/evaluate_model.py     # Evaluate the model
```

