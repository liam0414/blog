# Deep Learning II

Note: this week's contents are highly related to the project

## Lecture 1 and Lecture 2

### Object detection using CNNs

object detection vs instance segmentation

- object detection: bounding box around object
- instance segmentation: pixel level segmentation, we want to know which pixel belongs to which object

image classification: image -> CNN -> object class
object detection: image -> CNN -> bounding box -> object class

In object detection, we not only want to know the object class based on some probabilities, but also the location of the object in the image (bounding box) for each object.

**Intersection over Union (IoU)**
IoU is a measure of the overlap between two bounding boxes. It is defined as the area of overlap between the two bounding boxes divided by the area of union between the two bounding boxes.

#### Proposal based algorithms

##### R-CNN

- Extracts ~2000 region proposals from input image
- Computes CNN features for each proposal
- Classifies regions using SVMs

##### Fast R-CNN

- Passes entire image through CNN once
- Uses RoI pooling to extract features for proposed regions
- Single-stage training with multi-task loss

##### Faster R-CNN

- Introduces Region Proposal Network (RPN) to predict proposals
- Unifies region proposals with detection network
- Uses anchor boxes and non-maximum suppression

#### Proposal free algorithms

##### YOLO

- Divides image into grid and predicts bounding boxes and class probabilities for each cell
- Very fast but less accurate than proposal-based methods

##### SSD (Single Shot Detector):

- Uses multiple feature maps at different scales
- Predicts category scores and box offsets for default boxes

##### RetinaNet

- Uses Feature Pyramid Network
- Introduces Focal Loss to address class imbalance issue

### Semantic Segmentation

Semantic segmentation aims to classify every pixel in an image. Key techniques:

#### Fully Convolutional Networks (FCN):

Replaces fully connected layers with convolutional layers
Uses transposed convolutions for upsampling

#### U-Net:

Encoder-decoder architecture with skip connections
Widely used in biomedical image segmentation

#### U-Net variants:

- Attention U-Net
- ResUNet
- TransUNet (incorporating transformers)

### Instance Segmentation

Instance segmentation differentiates individual object instances. Key technique:

#### Mask R-CNN:

- Extends Faster R-CNN by adding a branch for predicting segmentation masks
- Uses RoIAlign for better feature alignment

### Video Understanding

Challenges in video processing include:

- Capturing information across frames
- High computational cost due to large data size

Key approaches:

- 3D CNNs: C3D architecture for spatiotemporal feature learning
- Two-Stream Networks: Separate streams for spatial (RGB) and temporal (optical flow) information
- Recurrent architectures: Long-term Recurrent Convolutional Networks (LRCN) for modeling long-term dependencies
- Transformer-based approaches: TimeSformer and ViViT for video classification using self-attention
