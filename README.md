# figure-skating-jumps-classifier

## Project Overview
Figure Skating Classifier using TensorFlow with Keras 

## Project Description:

Differentiating between the six figure skating jumps—Toe Loop, Salchow, Loop, Flip, Lutz, and Axel—can be challenging for newcomers to the sport. These jumps are often integrated into fast-paced routines, making visual differentiation even harder.

As a figure skating and AI/ML enthusiast, I developed a machine learning model utilizing TensorFlow and Keras to classify these jumps using 3D pose data from the FS-Jump3D dataset.


## Key differentiating factors include:

- Take-off Method: Identifying if the jump starts from a forward or backward position.

- Jump Type: Distinguishing between edge jumps (Salchow, Loop) and toe-pick jumps (Flip, Lutz, Toe Loop).

- Blade Edge: Noting whether the jump uses the inside or outside blade edge.

- Rotational Count: Accounting for the number of rotations, such as the Axel jump requiring an additional half rotation.


## Note: (for curious people :p)

- Salchow     -  Backward inside edge

- Loop        -  Backward outside edge

- Flip        -  Backward inside edge, with toe-pick

- Lutz        -  Backward outside edge, with toe-pick, landing on the opposite foot

- Toe Loop    -  Backward outside edge, with toe-pick, landing on the same foot

- Axel        -  Forward outside edge, with an additional half rotation in the air (a single Axel consists of one and a half rotation)


## Installation Instructions
To set up the environment, clone this repository and install the required packages:

```bash
git clone https://github.com/phoebenyi/figure-skating-jumps-classifier.git
cd figure-skating-jumps-classifier
python3 -m pip install -r requirements.txt
```

## Dataset Information
FS-Jump3D Dataset provides detailed 3D pose data of skaters performing different jumps and captures key body movements through markerless motion capture technology. This makes it ideal for analyzing complex 3D movements and subtle technical differences between jumps, such as take-off angles, rotation, and landing positions, which are critical for classifying each type of jump accurately.


## Running Instructions:
How to execute the scripts you’ve created.
