# Niryo Ned2 Conveyor Sorting

Vision-based object sorting using a Niryo Ned2, conveyor belt, IR object detection, an external USB camera, OpenCV, and K-Means color classification.

## Pipeline

Conveyor ON → IR detects object → Conveyor OFF → camera capture → object detection → shape + color classification → fixed Ned2 pick pose → target bin → release → conveyor ON.

The conveyor/IR station fixes the object's physical pick location, so the robot uses a calibrated fixed PICK_POSE rather than continuous camera-pixel-to-robot coordinate conversion.

## Project status

Initial repository structure is being built. Hardware-specific values such as robot IP, camera ID, serial port, pick pose, and bin poses must be calibrated for the physical setup.

## Planned components

- Python / OpenCV
- scikit-learn K-Means
- PyNiryo
- Arduino/ESP32 conveyor controller
- External USB camera
- Niryo Ned2
- Kaggle RGB color-space dataset for offline color-space training/reference
- Real camera images for physical-world evaluation

## Color classes

Red, Green, Blue, Yellow.

## Safety

Keep robot motion disabled while calibrating poses and verify all PICK_POSE and BIN_POSES at low speed before autonomous operation.
