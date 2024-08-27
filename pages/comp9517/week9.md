# Motion Estimation and Object Tracking

## Motion Estimation

### Introduction

- Adding the time dimension to image formation
- Analyzing changing scenes via image sequences
- Changes in image sequences provide features for:
  - Detecting moving objects
  - Computing trajectories
  - Performing motion analysis
  - Recognizing objects based on behaviors
  - Computing viewer motion
  - Detecting and recognizing activities

### Applications

- Motion-based recognition
- Automated surveillance
- Video indexing
- Human-computer interaction
- Traffic monitoring
- Vehicle navigation

### Scenarios

- Still camera
  - Constant background with single/multiple moving objects
- Moving camera
  - Relatively constant scene with coherent motion or moving objects

### Change Detection

- Detect moving objects against a constant background
- Steps:
  1. Derive background image
  2. Subtract background from each frame
  3. Threshold and enhance difference image
  4. Detect bounding boxes

### Sparse Motion Estimation

- Compute sparse motion field by identifying corresponding points in two images
- Steps:
  1. Detect interesting points (using edge/corner detectors, SIFT, etc.)
  2. Search for corresponding points in the next frame

### Dense Motion Estimation

- Optical Flow
- Assumptions:
  - Object reflectivity and illumination don't change
  - Distance to camera doesn't vary significantly
  - Small neighborhoods shift position between frames

#### Optical Flow Equation

- Derived from Taylor series expansion
- Constraint: $f_x * v_x + f_y * v_y + f_t = 0$
- Requires additional constraints for unique solution

#### Lucas-Kanade Approach

- Assumes constant flow in local neighborhood
- Solves system of equations using least squares

## Object Tracking

### Introduction

- Generating inference about object motion from image sequences

### Applications

- Motion capture
- Recognition from motion
- Surveillance
- Targeting

### Challenges

- Loss of information in 2D projection
- Image noise
- Complex object motion
- Non-rigid objects
- Occlusions
- Complex shapes
- Illumination changes
- Real-time requirements

### Bayesian Inference for Tracking

- Three main steps:
  1. Prediction
  2. Association
  3. Correction

#### Independence Assumptions

- Current state depends only on immediate past
- Measurements depend only on current state

#### Tracking Process

1. Prediction: $P(X_i | Y_0:i-1) = ∫ P(X_i | X_i-1) P(X_i-1 | Y_0:i-1) dX_i-1$
2. Correction: $P(X_i | Y_0:i) ∝ P(Y_i | X_i) P(X_i | Y_0:i-1)$

### Kalman Filtering

- Assumes linear models and Gaussian noise
- Prediction and correction steps with matrix operations

### Particle Filtering

- For non-linear/non-Gaussian cases
- Represents state density with weighted particles
- Propagates particles using dynamics model
- Updates weights using measurement model

### Applications

- Tracking active contours
- Object location tracking in clutter
