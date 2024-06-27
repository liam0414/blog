# Pattern Recognition

## Lecture 1

### Supervised Learning

Classification: SVM, Discriminant Analysis, Naive Bayes, Nearest Neighbour

#### Pattern recognition system

image acquisition -> pre-processing -> feature extraction -> feature selection -> learning system -> evaluation -> application

#### Key concepts

* Pre-processing: enhance images for further processing
* Feature extraction: reduce the data by measuring certain properties
* Feature descriptor: represent scalar properties of objects
* Feature vectors: capture all the properties measured from the data
* Feature selection: select the most relevant and important descriptive features
* Models: mathematical or statistical representation of classes
* Training samples: objects with known labels used to build models
* Cost: the consequence of making an incorrect decision/assignment
* Decision boundary: is the demarcation between regions in feature space

#### Pattern recognition example

Salmon vs Sea Bass

    Feature extraction: length, width, color, texture
    Feature selection: length, width (invariant to object position, orientation etc)
    Model: linear classifier
    Training samples: salmon and sea bass
    Cost: misclassification of salmon as sea bass
    Decision boundary: line separating salmon and sea bass

### Supervised learning overview

- Feature space X -> Label space Y
- Functions f: X -> Y
- Training set: \{(x1, y1), (x2, y2), ..., (xn, yn)\}
- Model: f(x) = y
- Cost function: C(f(x), y)
- Optimization: minimize C(f(x), y) over the training set

#### Two Pattern Recognition Models

**Generative models:**
- Model the joint probability distribution P(X, Y)
- Obtain the joint probability distribution P(X, Y) = P(X|Y)P(Y)
- Use Bayes rule to compute the posterior probability (decision boundary) $ P(Y|X) = P(X|Y)P(Y) / P(X) $

**Discriminative models:**

Model the conditional probability distribution P(Y|X)
Directly learn the decision boundary

#### Nearest Class Mean Classifier

the nearest class mean classifier is a simple generative model that classifies an object based on the distance to the mean of each class

training: given a set of labelled samples, compute the mean of each class
testing: given a new sample, compute the distance to the mean of each class and assign the class with the smallest distance

![NCMC](public/images/9517/week4/NCMC.png)

The problem with the nearest class mean classifier is that it assumes that the data is normally distributed and that the classes have the same covariance. This is not always the case in practice. For example, data may be skewed or have different variances. In such cases, the nearest class mean classifier may not perform well. So using the mean of the entire class to represent the class may not be the best approach.

nearest class mean classifier: works best when the data is normally distributed and the classes are far apart

#### K-Nearest Neighbor Classifier

For every test sample, the distance between the test sample and all training sample are computed. The k-nearest training samples are selected to decide the class label of the test sample. The class label is decided by majority voting. The neighbors are selected from a set of training samples for which the class labels are known.

**Hamming Distance and Euclidean Distance**

Euclidean distance is commonly used for continuous data, while Hamming distance is used for categorical data.

![hamming](public/images/9517/week4/hamming.png)

#### Bayesian Decision Theory

Bayesian classification assigns an object into the class to which it mostly likely belongs based on observed features.

- Assume the following to be known:
    - The prior probability of each class $ p(c_I) $
    - The class-conditional probability density function $ p(x|c_I) $

- Compute the posterior probability of each class 

$ p(c_I|x) =  \frac{p(x|c_I)p(c_I)}{\sum_{j=1}^{n} p(x|c_j)p(c_j)} $

- The error can be minimized by choosing the class with the highest posterior probability. The error is given by:

$ E = 1 - \sum_{i=1}^{n} p(c_i|x) $

**Example**
Salmon vs Seabass if we use prior experience to estimate the probability of each class, this can be wrong if the prior experience is not accurate.

| p(c_i) | salmon | seabass | other |
|--------|--------|---------|-------|
| Prior  | 0.3    | 0.6     | 0.1   |

Now we can use length and width to estimate the class-conditional

| $p(x\|c_i)$ | salmon | seabass | other |
|----------|--------|---------|-------|
| Length > 100cm  | 0.5    | 0.3     | 0.2   |
| 50 cm < Length < 100cm    | 0.4    | 0.5     | 0.2   |
| Length < 50cm  | 0.1    | 0.2     | 0.6   |

We can now compute the posterior probability of each class

$ p(c_i = salmon|x = 70cm) = p(70cm|salmon) * p(salmon) = 0.4 * 0.3 = 0.12 $

$ p(c_i = seabass|x = 70cm) = p(70cm|seabass) * p(seabass) = 0.5 * 0.6 = 0.3 $

$ p(c_i = other|x = 70cm) = p(70cm|other) * p(other) = 0.2 * 0.1 = 0.02 $

Note that the sum here is not 1, so we need to normalize the probabilities. Imagine p(70cm|salmon) = p(70cm|seabass) = 0.5, then we are essentially comparing the prior probabilities.

**Example continued**
If the price of salmon is twice that of sea bass, and sea bass is also more expensive than the other types of fish, is the cost of a wrong decision the same for any misclassification?

The answer is no, the cost of misclassifying a salmon as a sea bass is higher than the cost of misclassifying a sea bass as a salmon because the cost of a salmon is higher than the cost of a sea bass.

**Bayesian Decision Risk**

We want to minimize the loss from our prediction
$ R(a_i|x) = \sum_{j=1}^{n} \lambda(a_i|c_j) p(c_j|x) $

where $ R(a_i|x) $ is the conditional risk. The loss function $ \lambda(a_i|c_j) $ is the cost of taking action $ a_i $ when the true class is $ c_j $.

**Issue with Bayesian Decision Theory**

- The class-conditional probability density function $ p(x|c_i) $ is often unknown

- The prior probability $ p(c_i) $ is often unknown and hence subjective

#### Decision Trees

Decision trees are a popular supervised learning method used for classification and regression. They are simple to understand and interpret, and can handle both numerical and **categorical** data.

![decision_tree](public/images/9517/week4/decision_tree.png)

- Approach
    * classify a sample through a sequence of questions
    * next question asked depends on answer to current question
    * sequence of questions displayed in a directed decision tree

- Structure
    * nodes in the three represent features
    * leaf nodes contain the class labels
    * one feature at a time to split search space
    * each branching node has one child for each possible value of the feature

- Classification
    * begins at the root node
    * assign the class label of the leaf node

**Constructing Optimal Decision Tree**

We can use the entropy, defined as

$ H(S) = - \sum_{i=1}^{n} p_i log_2 p_i $

where $ p_i $ is the probability of class i in the set S.

to measure the impurity of a set of samples. The entropy is 0 if all samples in the set belong to the same class, and 1 if the samples are evenly distributed among the classes.

**Example**

| $p(c_i)$ | salmon | seabass | other |
|--------|--------|---------|-------|
| Prior  | 0.3    | 0.6     | 0.1   |

$ H = -0.3 log_2(0.3) - 0.6 log_2(0.6) - 0.1 log_2(0.1) = 1.29 $

#### Information Gain

Information gain is the measure of the effectiveness of a feature in classifying the samples. It is the difference between the entropy of the parent node and the weighted sum of the entropies of the child nodes.

$ IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v) $

where $ S_v $ is the subset of samples in S for which feature A has value v.

**Example**

| $x_1$ | $x_2$ | type |
|-------|-------|------|
| S     | S     | Salmon    |
| M     | S     | Salmon    |
| M     | S     | Salmon    |
| S     | M     | Sea bass    |
| S     | L     | Sea bass    |
| S     | M     | Sea bass    |
| M     | M     | Sea bass    |
| M     | L     | Sea bass    |
| L     | M     | Salmon    |
| L     | M     | Salmon    |
| L     | L     | Salmon    |
| S     | L     | Sea bass    |
| M     | L     | Sea bass    |
| M     | M     | Sea bass    |
| M     | L     | Sea bass    |

where $x_1 \isin \{S, M, L\}$ and $x_2 \isin \{S, M, L\}$

We can see from the above observations, #salmon = 6, #seabass = 9

So the probability of salmon is $ p(salmon) = 6 / 15 = 0.4 $ and the probability of seabass is $ p(seabass) = 9 / 15 = 0.6 $

We can calculate the base entropy as:
#H_base$ = -0.4 log_2(0.4) - 0.6 log_2(0.6) = 0.971

To estimate IG for $x_1$, we can calculate the entropy for each value of $x_1$ and then calculate the weighted sum of the entropies.

$H(S|x_1) = 5/15 H(1,4) + 7/15 H(2,5) + 3/15 H(3,0) = 0.64$

                    Salmon	Sea Bass		Weight	Total
    Entropy for S	0.20	0.8 	0.7219	0.3333	0.6434
    Entropy for M	0.29	0.71	0.8631	0.4667	
    Entropy for L	1.00	0.00	0.0000	0.2000	

$ IG = H_base - H(S|x_1) = 0.971 - 0.64 = 0.331 $

$H(S|x_2) = 3/15 H(3,0) + 6/15 H(2,4) + 6/15 H(1,5) = 0.62$

                    Salmon	Sea Bass		Weight	Total
    Entropy for S	1.00	0	    0.0000	0.2000	0.6273
    Entropy for M	0.33	0.67	0.9183	0.4000	
    Entropy for L	0.17	0.83	0.6500	0.4000	

$ IG = H_base - H(S|x_2) = 0.971 - 0.62 = 0.351 $

Since IG for $x_2$ is higher than IG for $x_1$, we choose $x_2$ as the root node. We then divide the dataset by its branches and repeat the same process on every branch. A branch with entropy more than 0 needs to be further divided.

### Ensemble Methods

Ensemble methods combine multiple models to improve the accuracy of the prediction. The idea is to train multiple models and combine their predictions to get a better result than any individual model.

**Random Forest**

Random forest is an ensemble method that uses multiple decision trees to improve the accuracy of the prediction. It creates a set of decision trees from randomly selected subset of the training set. It then combines the predictions of the individual trees to get the final prediction.

- Training:
    - Let N be number of training instances and M the number of features (--bagging)
    - Sample N instances at random with replacement from the original data
    - At each node, select m << M features are random and split on the best feature (--feature bagging)
    - Grow the tree to the largest extent possible
    - Repeat the above steps to create a forest of trees

- Testing:
    - Push a new sample down a tree and assign the label of the leaf node
    - Iterate over all trees to get B predictions for the sample
    - Report the majority vote of all trees as the random forest prediction

- Key concepts:
    - the random forest error rate depends on two factors:
        * the correlation between any two trees in the forest
        * the strength of each individual tree
    - selecting parameter m
        * reducing m reduces both the correlation between trees and the strength of each tree
        * increasing m increases the correlation between trees and the strength of each tree

**Sample Exam Question**
Which one of the following statements is correct about random forests classifiers?

* A. Increasing the correlation between the individual trees decreases the random forest classification error rate
* B. Reducing the number of selected features at each node increase the correlation between the individual trees
* C. Reducing the number of selected features at each node increases the strength of the individual trees
* D. Increasing the strength of the individual trees decreases the random forest classification error rate.

A. Increasing the correlation between the individual trees decreases the random forest classification error rate.

    This is incorrect. In random forests, having highly correlated trees can lead to similar mistakes being made by the trees, which does not significantly reduce the overall error. Reducing the correlation between the trees helps in making the ensemble more robust and improves the overall performance.

B. Reducing the number of selected features at each node increases the correlation between the individual trees.

    This is incorrect. Reducing the number of features selected at each node actually decreases the correlation between the individual trees, as it forces each tree to consider different subsets of features, leading to more diverse trees.

C. Reducing the number of selected features at each node increases the strength of the individual trees.

    This is incorrect. Reducing the number of features at each node usually decreases the strength (individual accuracy) of each tree because each tree has less information to make a decision at each split. However, it increases the diversity among trees.

D. Increasing the strength of the individual trees decreases the random forest classification error rate.

    This is correct. The strength of an individual tree refers to its accuracy. If the individual trees are more accurate (stronger), then the overall performance of the random forest improves, reducing the classification error rate.

## Lecture 2

Regression: Linear Regression, SVR, GPR, Ensemble Methods, Decision Trees, Neural Networks

**Separability**

- Linearly separable: can be separated by a straight line
- Non-linearly separable: cannot be separated by a straight line

### Linear Classifier

Given a training set of N observations:
$ D = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\} $

where $ x_i $ is the feature vector and $ y_i $ is the class label.

A binary classification problem can be modelled by a separation function f(x) such that:

$$
f(x_i) = 
\begin{cases}
      > 0 & \text{if $y_i$ = 1}\\
      < 0 & \text{if $y_i$ = -1}\\
\end{cases}
$$

A generalised form of the separation function is:
$$ f(x) = w^T x + b = w_1x_1 + w_2x_2 + ... + w_dx_d + b $$

where $x_i$ are features, w is the weight vector and b is the bias term.

### Support Vector Machines (SVMs)

Maximize margin - the distance between the decision boundary and the nearest data point from either class. The decision boundary is the hyperplane that separates the classes.

Why do we want to maximise margin? This is because if we are too close to the data points, our classifier is more likely to make mistakes. By maximizing the margin, we are ensuring that our classifier is more robust and less likely to make mistakes.

The primal optimization problem for linear SVM (Hard-margin SVMs) is:

$$
\text{min} \frac{1}{2} ||w||^2_2 \\
\text{subject to } y_i(w^T x_i + b) \geq 1, i = 1, 2, ..., N
$$

Decision rules in testing:
$$
f(x) =
\begin{cases}
      1 & \text{if $w^T x + b > 0$}\\
      -1 & \text{if $w^T x + b < 0$}\\
\end{cases}
$$

**Hard-margin SVMs**

- Assumes that the data is linearly separable
- Does not allow for any misclassification
- Sensitive to outliers

**Soft-margin SVMs**

In practice, data is not always linearly separable. Soft-margin SVMs allow for some misclassification by introducing slack variables. The primal optimization problem for soft-margin SVMs is:

$$
\text{min} \frac{1}{2} ||w||^2_2 + C \sum_{i=1}^{N} \xi_i \\
\text{subject to } y_i(w^T x_i + b) \geq 1 - \xi_i, i = 1, 2, ..., N \\
\xi_i \geq 0, i = 1, 2, ..., N
$$

where C is the regularization parameter that controls the trade-off between maximizing the margin and minimizing the misclassification.

>Compare and contrast the hard-margin and soft-margin SVMs:

- Hard-margin SVMs:
    * Assumes that the data is linearly separable
    * Does not allow for any misclassification
    * Sensitive to outliers

- Soft-margin SVMs:
    * Allows for some misclassification
    * Introduces slack variables to handle misclassification
    * Uses a regularization parameter C to control the trade-off between maximizing the margin and minimizing the misclassification

**Kernel Trick**

The kernel trick is a method used to transform the input data into a higher-dimensional space without explicitly computing the transformation. This allows SVMs to handle non-linearly separable data by finding a hyperplane in the higher-dimensional space that separates the classes.

![kernel_trick](public/images/9517/week4/kernel_trick.png)

### Multi-class Classification

Some methods may be directly used for multiclass classification, while others (such as SVM) may need to be modified or combined to handle multiple classes.

- K-Nearest Neighbour
- Decision Trees
- Neural Networks

**However there are two possible techniques for multiclass classification with binary classifiers:**

1. One-vs-All (OvA) or One-vs-Rest (OvR): Train a binary classifier for each class, where the class is treated as the positive class and all other classes are treated as the negative class. During testing, the class with the highest confidence score is assigned as the predicted class.

2. One-vs-One (OvO): Train a binary classifier for each pair of classes. During testing, each classifier votes for one of the two classes, and the class with the most votes is assigned as the predicted class.

### Evaluating Models

- Type I error: false (alarm) positive
- Type II error: false (dismissal) negative

- Accuracy: the proportion of correctly classified instances

$$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} $$

- Precision: the proportion of true positive predictions among all positive predictions

$$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}} $$

- Recall: the proportion of true positive predictions among all actual positive instances

$$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}} $$

- F1 Score: the harmonic mean of precision and recall

$$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}} $$

- Confusion Matrix: a table that shows the number of true positives, true negatives, false positives, and false negatives

- Receiver Operating Characteristic (ROC) Curve: a graphical plot that illustrates the performance of a binary classifier at various threshold settings

![roc](public/images/9517/week4/roc.png)


**Example**

Consider a binary classification problem with the following confusion matrix:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | 100 | 20 |
| Actual Negative | 10 | 70 |

Calculate the accuracy, precision, recall, and F1 score.

- Accuracy: (100 + 70) / (100 + 20 + 10 + 70) = 0.85
- Precision: 100 / (100 + 10) = 0.91
- Recall: 100 / (100 + 20) = 0.83
- F1 Score: 2 * (0.91 * 0.83) / (0.91 + 0.83) = 0.87

### Regression

Regression is a supervised learning technique used to predict continuous values. It models the relationship between the input features and the target variable.

#### Linear Regression

Linear regression is a simple regression model that assumes a linear relationship between the input features and the target variable.

$$ y = w^T x + b $$

where y is the target variable, x is the input features, w is the weight vector, and b is the bias term.

The goal of linear regression is to find the optimal values of w and b that minimize the error between the predicted values and the actual values.

#### Least Squares Method

The least squares method is a common approach to solving linear regression problems. It minimizes the sum of the squared differences between the predicted values and the actual values.

$$ \text{min} \sum_{i=1}^{N} (y_i - w^T x_i - b)^2 $$

The optimal values of w and b can be found by solving the normal equations:

$$ w = (X^T X)^{-1} X^T y $$

where X is the matrix of input features and y is the vector of target values.

#### Regression Evaluation Metrics

- Mean Squared Error (MSE): the average of the squared differences between the predicted values and the actual values

$$ \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y_i})^2 $$

- Root Mean Squared Error (RMSE): the square root of the MSE

$$ \text{RMSE} = \sqrt{\text{MSE}} $$

- Mean Absolute Error (MAE): the average of the absolute differences between the predicted values and the actual values

$$ \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y_i}| $$

- R-squared (R2): a measure of how well the regression model fits the data

$$ R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y_i})^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2} $$

where $\bar{y}$ is the mean of the target values.

#### Normalization on features

Normalization is the process of scaling the input features to a similar range. This can help improve the performance of the regression model by ensuring that all features contribute equally to the prediction.

Common normalization techniques include min-max scaling and z-score normalization.

- Min-Max Scaling: scales the input features to a range between 0 and 1

$$ x_{\text{norm}} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)} $$

- Z-Score Normalization: scales the input features to have a mean of 0 and a standard deviation of 1

$$ x_{\text{norm}} = \frac{x - \text{mean}(x)}{\text{std}(x)} $$

#### Cross-Validation

Cross-validation is a technique used to evaluate the performance of a machine learning model. It involves splitting the data into multiple subsets, training the model on one subset, and testing it on another subset.

Common cross-validation techniques include k-fold cross-validation and leave-one-out cross-validation.

- K-Fold Cross-Validation: the data is divided into k subsets, and the model is trained and tested k times, each time using a different subset as the test set.

- Leave-One-Out Cross-Validation: each data point is used as the test set once, and the model is trained on the remaining data points.

**Sample Exam Question**

Which one of the following statements is correct for pattern recognition?

A. Pattern recognition is defined as the process of model training on a training dataset and then testing on an independent test set.

B. The dimension of feature vectors should be smaller than the number of training samples in order to avoid the overfitting problem.

C. The simple kNN classifier needs homogeneous feature types and scales so that the classification performance can be better.

D. SVM is a powerful classifier that can separate classes even when the feature
space exhibits significant overlaps between classes.

    The answer is C. The simple kNN classifier needs homogeneous feature types and scales so that the classification performance can be better.

    This is because kNN (k-Nearest Neighbors) relies on calculating distances between data points, and having features with different types or scales can lead to misleading distance calculations, affecting the classification performance.

    A. Pattern recognition is not solely defined by the process of model training on a training dataset and then testing on an independent test set. While this is a common approach in evaluating models, pattern recognition encompasses a broader set of techniques and processes for identifying patterns in data.

    B. The dimension of feature vectors being smaller than the number of training samples can help in some cases, but it is not a strict rule to avoid overfitting. Overfitting can also be managed through techniques such as regularization, cross-validation, and using more sophisticated models.

    D. SVM (Support Vector Machine) is powerful but generally works best when classes are linearly separable or can be made linearly separable in a higher-dimensional space through the kernel trick. When there is significant overlap in the feature space, SVM might struggle unless the overlap is manageable through the choice of an appropriate kernel.