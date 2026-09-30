# Niryo Ned2 Conveyor Sorting

Vision-based sorting using a Niryo Ned2, conveyor, IR stop sensor, external USB camera, OpenCV and K-Means colour classification.

## System architecture

    CONVEYOR ON
        ↓
    IR detects object at fixed station
        ↓
    CONVEYOR OFF
        ↓
    settling delay
        ↓
    USB camera captures fixed ROI
        ↓
    OpenCV detects object + extracts shape
        ↓
    K-Means predicts RED/GREEN/BLUE/YELLOW
        ↓
    Ned2 moves to calibrated fixed PICK_POSE
        ↓
    gripper picks object
        ↓
    colour-specific BIN_POSE
        ↓
    release
        ↓
    CONVEYOR ON

## Why there is no camera-to-robot homography

The IR sensor stops every object at the same physical pick station. The camera is therefore used for classification and verification, while the robot uses one calibrated physical PICK_POSE.

## ML methodology

The Kaggle RGB dataset represents the RGB colour space. It is used as an offline colour-space reference/training source rather than as a replacement for physical camera images.

1. Sample RGB values.
2. Assign RED, GREEN, BLUE and YELLOW using the project colour rules.
3. Derive normalized RGB and HSV features.
4. Standardize the features.
5. Train K-Means with 4 clusters.
6. Map cluster IDs to semantic colour labels.
7. Evaluate on held-out samples.
8. Separately collect real USB-camera images under the actual conveyor lighting for physical-world evaluation.

K-Means cluster IDs have no semantic meaning by themselves, so color_mapping.json converts cluster IDs to colour names.

## Setup

Install dependencies:

    pip install -r requirements.txt

Place the downloaded Kaggle ZIP in dataset/raw/.

Prepare a compact dataset:

    python -m src.dataset.prepare_rgb_dataset --zip dataset/raw/rgb_colour_dataset.zip --output dataset/processed/rgb_training.csv --samples 250000

Or sample the RGB space directly:

    python -m src.dataset.prepare_rgb_dataset --grid --output dataset/processed/rgb_training.csv --samples 250000

Train:

    python -m src.training.train_color_model --csv dataset/processed/rgb_training.csv

Evaluate:

    python -m src.training.evaluate_color_model --csv dataset/processed/rgb_validation.csv

## Hardware configuration

Edit config.py before running:

- ROBOT_IP
- CAMERA_ID
- CONVEYOR_SERIAL_PORT
- PICK_POSE
- BIN_POSES
- ROI
- MIN_CONTOUR_AREA

Do not copy pose coordinates from another robot/setup. Calibrate them on the actual Ned2 and conveyor.

## Arduino

Upload arduino/conveyor_controller/conveyor_controller.ino.

Python sends START and STOP.
Arduino sends OBJECT_DETECTED.

Change motor-driver pins, IR logic and PWM to match the selected hardware.

## Commissioning order

1. Test conveyor alone.
2. Test IR sensor and serial messages.
3. Test USB camera and ROI.
4. Test object detection without the robot.
5. Test colour classification with real objects.
6. Test Ned2 with no object at low speed.
7. Verify PICK_POSE manually.
8. Verify every BIN_POSE manually.
9. Run one-object-at-a-time sorting.
10. Enable continuous operation only after the above tests pass.

## Limitations

- The Kaggle dataset contains solid RGB colour information and does not model all physical lighting/material effects.
- Real camera images should be the main physical-world test set.
- Lighting, shadows, reflections, exposure, object material, background and ROI positioning affect results.
- Shape classification is currently heuristic based on contour geometry.
- The current robot wrapper assumes a gripper. A vacuum tool requires changing the pick/place tool commands.

## Status

Initial end-to-end software skeleton is committed. Hardware calibration, real-world image collection, trained model artifacts and final validation remain to be added.
