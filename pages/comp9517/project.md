# Project Specification:

Group project: worth 40% of the total course mark

Task: Develop and compare different computer vision methods for semantic segmentation of natural environment images

Dataset: WildScenes, containing 9,306 images from two Australian forests

Requirements:

- Develop at least two different methods
- Use training, validation, and test splits as specified
- Evaluate performance using Intersection over Union (IoU)
- Deliverables: 10-minute video presentation, written report (max 10 pages), and source code

## Our Group: 40/40

The marking heavily depends on two things: the design of loss function and regularization to prevent over-fitting, and hyperparameter tuning (huge marks)

Developed three baseline models: U-Net, U-Net++, SegNet, and YOLOv8

Implemented data preprocessing techniques, including mean subtraction and hyperparameter tuning

Applied post-processing methods such as superpixels, conditional random fields (CRF), pixel and morphological operations, and ensemble learning

Provided a well-organized folder structure with main notebooks (eda.ipynb and model_run.ipynb) and subfolders for models, notebooks, and dataset

Included detailed instructions for setting up the environment and running the code

Discussed challenges in natural scene segmentation and proposed future work
