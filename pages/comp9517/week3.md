# Feature Representation

## Lecture 1

### Image Feature

- Image features are often vectors that are a compact representation of images, they represent important information in the image.
    * Object detection: identify the location of an object in an image and what the object is
    * Image segmentation: dividing an image into regions (binary segmentation), identify whether a pixel is an object or background
    * Image classification: assign a label to an image
    * Image retrieval: find similar images in a database
    * Image stitching: combine multiple images into a single image
    * Object tracking: track an object in a video

#### Why do we need image features?

If we take pixel values directly as features, 
    * the feature vector will be very large, it will be difficult to process
    * pixels can be redundant
    * pixels are not invariant to translation, rotation, scale, etc.
So we need to extract features that are more compact and informative.

### Categories of Image Features

Desirable properties of features including:
    * Reproducibility (robustness): should be detectable at the same location under different conditions such as illumination and viewpoint
    * Saliency (descriptiveness): similar salient points in differnet images should have similar features
    * Compactness (efficiency): should be able to represent the image with a small number of features

### Colour Features

##### Colour Histograms

The colour histogram represents the global distribution of pixel colours in an image. It is very simple idea
    - Step 1: construct a histogram for each colour channel (R, G, B)
    - Step 2: concatenate the histograms to form a single feature vector

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
g_hist = cv2.calcHist([img], [1], None, [256], [0, 256])
b_hist = cv2.calcHist([img], [2], None, [256], [0, 256])

plt.plot(r_hist, color='red')
plt.plot(g_hist, color='green')
plt.plot(b_hist, color='blue')
plt.show()

# concatenate the histograms
hist = np.concatenate([r_hist, g_hist, b_hist])
print(hist)

```

##### Colour Moments

Colour moments are statistical measures of the distribution of pixel intensities in an image.
    - First-order moments: mean of each colour channel
    - Second-order moments: variance and covariance of each colour channel
    - Third-order moments: skewness of each colour channel

Moments based representation of colour distributions
    - Gives a feature vector of only 9 elements (3 means, 3 variances, 3 covariances)
    - Lower representation capability compared to histograms

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mean, std_dev = cv2.meanStdDev(img)
cov = np.cov(img.reshape(-1, 3).T)

feature = np.concatenate([mean, std_dev, cov.diagonal()])
print(feature)
```

**We can combine colour, texture and shape features to get a more robust representation of the image**

```python
# only colour and texture
import cv2
import numpy as np

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# colour features
mean, std_dev = cv2.meanStdDev(img)
cov = np.cov(img.reshape(-1, 3).T)
colour_feature = np.concatenate([mean, std_dev, cov.diagonal()])
print(colour_feature)

# texture features (Haralick)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
haralick = cv2.HuMoments(cv2.moments(gray)).flatten()
lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
texture_feature = np.concatenate([haralick, lbp])
print(texture_feature)

# combine all features
feature = np.concatenate([colour_feature, texture_feature, shape_feature])
print(feature)
```

### Texture Features

#### Haralick Features

Haralick features are arrays of statistics calculated from the grey-level co-occurrence matrix (GLCM) of an image.
    - Step 1: Construct the gray level co-occurrence matrix (GLCM)
        * given distance d and orientation angle $ \theta $, the GLCM is a matrix where each element (i, j) represents the number of times a pixel with intensity i occurs at a distance d and orientation theta from a pixel with intensity j
        * compute co-occurrence count $ p_{(d, \theta)} $ of going from gray level $ i_1 $ to $ i_2 $ at distance $ d $ and orientation $\theta$
        * construct matrix $ P_{(d, \theta)} $ with elements $ (i_1, i_2) $ being $ p_{(d, \theta)} (i_1, i_2) $
        * If an image has L distinct gray levels, the matrix size is $ L \times L $

        ![haralick](public/images/9517/haralick.png)

```markdown
Notes for step 1:
    - if we have a L = 256 gray level image, then the matrix is going to be 256 x 256, 
      this is going to be a problem, we can use a smaller number n to bin the pixel intensities,
      for example, we can divide the pixel intensities by 16 and taking the floor of the result.

    - different co-occurrence matrices can be constructed by using various combinations of disntance and angle,
      this needs trial and error to find the best combination

    - On their own these matrices are not very useful, we don't use them directly

    - The information in the co-occurrent matrices needs to be further extracted as a 
      set of feature values such as Haralick descriptors
```

    - Step 2: Calculate the statistics from the GLCM (contrast, correlation, energy, homogeneity)
        * Haralick provides 14 features in total
        * These features are calculated from the GLCM

**Application:**
    1. Preprocess the biparametric MRI images
    2. Extract the Haralick features, run length and histogram features from the preprocessed images
    3. Apply feature selection to select the most important features
    4. Train a classifier to classify the images (KNN for instance)

#### Local Binary Patterns

Local binary patterns (LBP) are a simple yet powerful texture descriptor.
    - Step 1: Divide the image into N x N cells
    - Step 2: For each pixel in the cell, compare the pixel value with its 8 neighbours pixels
    - Step 3: If the centre pixel is greater than the neighbour, assign 0, otherwise assign 1
    - Step 4: This gives an 8-bit binary number pattern per pixel
    - Step 5: Convert the binary number to a decimal number
    - Step 6: Construct a histogram of the decimal numbers for each cell
    - Step 7: Concatenate the histograms to form a feature vector

    > for below example, the orientation starts from 2 go counter-clockwise 2 -> 3 -> 2 -> 2 -> 0 -> 0 -> 0 -> 0

    ![local_binary](public/images/9517/local_binary.png)

    ![local_binary2](public/images/9517/local_binary2.png)

**Important Concept**
We want invariant features, to achieve this, we can use bit shifting (shift by 1 bit). And what we use in the end is the minimum of the 8 shifted values.
So in whatever orientation we are, we will always get the same value, which is the minimum of the 8 shifted values.

#### Scale-Invariant Feature Transform (SIFT)

 [Link](https://docs.opencv.org/4.6.0/da/df5/tutorial_py_sift_intro.html) here to understand SIFT 

SIFT feature describes texture in a localised region around a keypoint. SIFT descriptor is invariant to various transformations
Recognising the same object inrespect to scaling, rotation, affine distortion, illumination changes.

**Four step process:**

##### 1. Scale-space extrema detection

Find local maxima and minima in the difference of Gaussian (doG) in the scale space (3D)

$ L(x, y, \sigma) = G(x, y, \sigma) * I(x, y) $ where $ G(x, y, \sigma) $ is a Gaussian kernel and $ I(x, y) $ is the image.

To create the stack, $ D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) $ where $ k $ is a constant

In the stack we built, we find the local maxima and minima in the 3D space. That means, we need to look at the level above and below the current level.

##### 2. Keypoint localisation

From step 1, we have a list of potential keypoints, we need to localise these keypoints more accurately.
    - Fit a 3D quadratic function to the DoG function at each keypoint to get subpixel optima
    - If we only do first step, we will have a lot of irrelevant keypoints, so we need to reject low-contrast and edge points using the Hessian matrix

##### 3. Orientation assignment

Estimate keypoint orientation to achieve invariance to image rotation.
    - Make an orientation histogram of local gradient vectors
    - Find the dominant orientation from the main peak in the histogram
    - Create additional keypoints for second highest peak if it is above a threshold (i.e., 80% of the main peak)

##### 4. Keypoint descriptor

Represent the local image region around the keypoint with a feature vector
    - Divide the region around the keypoint into 4 x 4 subregions
    - For each subregion, create an 8-bin histogram of gradient orientations
    - Concatenate the histograms to form a 128-dimensional feature vector

Using the nearest neighbour distance ratio (NNDR)
    - For each keypoint in the first image, find the two closest keypoints in the second image
    - If the ratio of the distance between the closest and second closest keypoints is less than a threshold, the match is considered correct

#### Application Example: Image stitching

Image stitching is the process of combining multiple images into a single image
    - Detect keypoints in each image
    - Compute the SIFT descriptor for each keypoint

The question is, how do we match the keypoints between images?

##### 1. Least-Squares Matching (LSM)

Least-squares matching is a robust method for matching keypoints between images
    - For each keypoint in the first image, find the closest keypoint in the second image
    - Compute the transformation that aligns the keypoints in the first image with the keypoints in the second image
    - If the distance between the transformed keypoints is less than a threshold, the match is considered correct

To find the least sequare, we take the pseudo inverse of the matrix and multiply it by the vector.

$ p = (A^T A)^{-1} A^T b $

> problem with this approach is outliers, we are essentially squaring the error, so if we have a few outliers, it will have a big impact on the result.

##### 2. Random Sample Consensus (RANSAC line fitting model)

RANSAC is a robust method for estimating the transformation that aligns keypoints between images
    - Sample (randomly) the number of points required to estimate the transformation
    - Solve for the model parameters using the sampled points
    - Score by the fraction of inliers within a present threshold
    - Repeat the above steps until a good result with high confidence

**Sample Exam question**

Which one of the following statements about feature descriptors is incorrect?
    - A. Haralick features are derived from gray-level co-occurrence matrices
    - B. SIFT achieves rotation invariance by computing gradient histograms at multiple scales.
    - C. LBP describes local image texture and can be multiresolution and rotation invariant.
    - <font color="red">D. Colour moments have lower representation capability than the colour histogram.</font>
Explanation:

Colour moments (mean, variance, and skewness of color channels) can capture essential color distribution information 
with fewer features, offering a more compact and often equally descriptive representation compared to the colour histogram, 
which may require a larger number of bins to achieve the same level of detail. Hence, colour moments do not necessarily 
have lower representation capability; in fact, they are often more efficient and effective for certain tasks.

## Lecture 2

### Another application of SIFT (Classification)

To classify images based on texture:
    - The number of SIFT keypoints may vary highly
    - Thus the number of SIFT descriptor may vary
    - Distance calculations require equal numbers

>To solve this problem, we can use the feature encoding technique called Bag of Visual Words (BoVW). Basically the number is to encode the features into a fixed-dimensional histogram

![bow](public/images/9517/bow.png)

#### Steps to create a BoVW model
    - Step 1: Extract SIFT descriptors from the training images
    - Step 2: Cluster the SIFT descriptors into K clusters using K-means (unsupervised method)
    - Step 3: Assign each SIFT descriptor to the nearest cluster
    - Step 4: Construct a histogram of the cluster assignments for each image
    - Step 5: Train a classifier on the histograms


```python
import cv2
import numpy as np

# Step 1: Extract SIFT descriptors from the training images
sift = cv2.SIFT_create()
descriptors = []
for img in images:
    kp, des = sift.detectAndCompute(img, None)
    descriptors.append(des)

# Step 2: Cluster the SIFT descriptors into K clusters using K-means
descriptors = np.concatenate(descriptors, axis=0)
kmeans = cv2.KMeans(n_clusters=K).fit(descriptors)

# Step 3: Assign each SIFT descriptor to the nearest cluster
histograms = []
for des in descriptors:
    labels = kmeans.predict(des)
    hist, _ = np.histogram(labels, bins=K, range=(0, K))
    histograms.append(hist)

# Step 4: Construct a histogram of the cluster assignments for each image
histograms = np.array(histograms)

# Step 5: Train a classifier on the histograms
```

#### k-means clustering

- initialize: k cluster centres (typically randomly)
- iterate:
    * assign each data point to the nearest cluster centre
    * update each cluster centre to the mean of the data points assigned to it
- Terminate when the cluster centres do not change significantly

```python
# implementation of k-means clustering
# initialize: k cluster centres (typically randomly)
k = 3
centres = np.random.rand(k, 2)

# iterate
while True:
    # assign each data point to the nearest cluster centre
    labels = np.argmin(np.linalg.norm(data[:, None] - centres, axis=2), axis=1)
    # update each cluster centre to the mean of the data points assigned to it
    for j in range(k):
        centres[j] = np.mean(data[labels == j], axis=0)
    # Terminate when the cluster centres do not change significantly
    if np.allclose(centres, new_centres):
        break
```

![feature_encoding](public/images/9517/feature_encoding.png)

> Feature encoding is not just for SIFT, it can be used for other features as well, for example local features (LBP, SURF, BRIEF, ORB...)

> There are also more advnaced feature encoding techniques such as Fisher Vector, VLAD, etc.

### Shape Features

Shape features are used to describe the shape of objects in an image, but we have challenges
    - invariant to right transformation, for example, translation, rotation, scaling
    - tolerant to non-rigid deformations, for example, bending and stretching
    - unknown correspondence between shapes, for example, different poses of the same object

#### Basic Shape Features

Convex vs Concave:
    - Convex: all points on the line segment connecting two points in the shape are also in the shape
    - Concave: not all points on the line segment connecting two points in the shape are in the shape

Simple geometrical shape descriptors:
    - Circularity: ratio of the area to the square of the perimeter, anything that is not a circle will have a circularity less than 1
    - Compactness: ratio of the area to the perimeter
    - Elongation: ratio of the major axis length to the minor axis length
    - Eccentricity: ratio of the distance between the foci to the major axis length

#### Boundary descriptors

Chain code descriptor: encode the boundary of the shape as a sequence of directions.
    - Start at a point on the boundary
    - Move to the next point in the clockwise direction
    - Encode the direction of the movement as a number (0-7)
    - Repeat until the starting point is reached

![chain_code](public/images/9517/chain_code.png)

Local curvature descriptor: encode the curvature of the boundary at each point
    - Compute the curvature at each point on the boundary
    - Encode the curvature as a number
    - Convex points have positive curvature, concave points have negative curvature

Global curvature descriptor: encode the curvature of the entire boundary
    - Compute the curvature at each point on the boundary
    - Compute the mean and standard deviation of the curvature

#### Example application of shape features

We are building the future space using area and circularity:
    - On Y axis, we have the area of the shape
    - On X axis, we have the circularity of the shape
    - We can use this to classify the shapes

![shape_descriptor](public/images/9517/shape_descriptor.png)

### Shape Context

Shape context is a **point-wise** local feature descriptor for shape matching
    - Step 1: Divide the boundary into N points
    - Step 2: For each point, compute the distance and angle to all other points
    - Step 3: Construct a histogram of the distances and angles
    - Step 4: Concatenate the histograms to form a feature vector

#### Example application of shape context

We want to match two shapes, we can use shape context to do this
    - Step 1: Sample a list of points on shape edges: Canny edge detection, Gaussian filtering -> intensity gradient -> non-maximum suppression -> hysteresis thresholding -> edge tracking
    - Step 2: Compute the shape content for each point
    - Step 3: Compare the **cost matrix** between two shapes P and Q
    - Step 4: Find the one to one matching minimising the total cost using the Hungarian algorithm
    - Step 5: Transform one shape to the other based on the one-to-one matching (for example affine, apply least squares or RANSAC fitting)
    - Step 6: Compute the shape distance


#### Histogram of Oriented Gradients (HOG)

Histogram of oriented gradients (HOG) is a feature descriptor for object detection
    - Step 1: Compute the gradient magnitude and orientation of each pixel
    - Step 2: Construct the gradient histogram of all pixels in a cell
    - Step 3: Generate detection-window level HOG descriptor.
    - Step 4: Concatenate the histograms to form a feature vector

> we need to try different hyperparameters to get the best result, for example, cell size, block size, detection window size, etc.

### Summary

- descriptor matching: SIFT, BoVW
- least squares matching: RANSAC
- spatial transformation: scale, shear, rotate, translate, affine, perspective
- feature encoding: BoVW, Fisher Vector, VLAD
- K-means clustering: unsupervised learning
- shape matching: shape context, HOG
- sliding window: object detection

**Exam Question**

Given the image on the right showing the result of a segmentation of various object and desired classification of these objects.
The two different colours (red and green) indicate the two different classes which the object are to be assign to. A straightforward way to perform
classification is by computing the value of a quantitative shape measure of each object and then thresholding those values. Suppose we
compute the circularity and eccentricity, which of these two measures can be used to separate the two classes?

![question](public/images/9517/question.png)

* <font color="red">A. Circularity</font>
* B. Eccentricity
* C. Both
* D. Neither

Explanation: Circularity is defined as the ratio of the area to the square of the perimeter.
In this question , the red objects are more circular than the green objects, so we can use circularity to separate the two classes.
The major and minor axes are perpendicular to one another, with the major axis pointing in the direction where the object's mass is most widespread, 
and the minor axis in the direction where it is most compact. For the object in question, the lengths of these axes are pretty much the same, so the eccentricity is 1