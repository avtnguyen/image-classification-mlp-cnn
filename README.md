# Image Classification with MLP and CNN
<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Photos.jpg"  width="300">

#### -- Project Status: [In Progress]

## Project Objective
The purpose of this project is to build a machine learning model for image classification using two deep learning techniques: the multilayer perceptron (MLP) and the convolutional neural network (CNN). The MNIST data is  a large database of handwritten digits and is available in tensorflow dataset.

## Project Description
**About the dataset** The MNIST dataset contains 60,000 training images and 10,000 testing images. Each image is 28x28 pixel.

Train data shape: (60000, 28, 28) (60000,)

Test data shape: (10000, 28, 28) (10000,)

<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Train_Test_classes.png">

The first 25 image in training dataset:

<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Handwriting_image.png">

### Project Pipeline :
**1. Data preprocessing for MLP and CNN model:** 
* One-hot encoding to convert a class vector to a binary class matrix
* Flatten images and normalization 

**2. Build a MLP and CNN models:**

**MLP model architecture**
* Three layers: input with 784 neurons, hidden layer with 128 neurons, and output layer with 10 neurons. This is approximated based on the number of features and number of labels.
* Dropout layer is added to in first and second layer to regularize the model and prevent overfitting. The dropout rate is assumed to 0.4
* Rectified Linear Unit (ReLU) is implemented in first and second layers which can help reduce the vanishing gradient problems. 
* For the last layer, softmax is used to return an array of probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes.
* Loss function used in here is the categorical cross-entropy to compute the error between the true classes and predicted classes. 
* Optimizer used here is the adam (adaptive moments) with adaptive learning rates
* The labels are given in an one_hot format
<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/MLP_model_plot.png">

**Network hyperparameter**
* Batch size: number of samples to work through before updating the internal model parameters. Smaller batch size will cause a longer training time but can prevent the overfitting issue since only part of the training dataset is seen. Common values equal to  2n  such as 32, 64, 128
* Epoch: number times that the learning algorithm will work through the entire training dataset. Increase number of epoch will improve the model but can lead to the overfitting problems
* Number of neurons per layer and number of hidden layer: Here, there will be only 1 hidden layer and the input layer will have 784 neurons corresponding to the input size.
* Dropout layer is a regularization method to turned off some neurons randomly with the dropout rate assumed to be 40%, meaning that there are 40% chances turned off randomly

**3. Model evaluation**

**Loss function versus epoch**

* Loss function vs epochs 

<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Los_vs_epoch_MLP.png">

**Confusion matrix**
* Confusion matric in MLP model
<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Confusion_Matrix_MLP.png">

**Classification report**
* Classification report for MLP model
<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/classification_report_MLP.png">

### Results:
For MLP model:
1. The accuracy of the model is 98% with good prediction for the handwriting of number 0,1,2,6 and a slightly less accurate prediction for numbers 3, 4, 5 ,8 and 9.
2. The lowest precision score is for number 9 and lowest recall scores are for number 8 and number 4. 

### Methods Used
* Data Visualization
* Data preprocessing for image classification
* Machine Learning Model: MLP and CNN

### Technologies
* Pandas
* Numpy
* Seaborn and Pyplot
* sklearn
* TensorFlow
* Colab

## Needs of this project
- Data processing
- Data modeling
- Writeup/reporting

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
3. Data processing and modeling scripts are being kept [here](https://github.com/avtnguyen/image-classification-mlp-cnn)

## References:
* https://deepai.org/machine-learning-glossary-and-terms/feed-forward-neural-network

## Contributing Members

**Team Leads (Contacts) : [Anh Nguyen ](https://github.com/avtnguyen)**

## Contact
* Feel free to contact team leads with any questions or if you are interested in contributing!
