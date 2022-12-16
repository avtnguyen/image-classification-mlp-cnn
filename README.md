# Image Classification with MLP and CNN

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

* Loss function vs epochs in MLP model

<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Los_vs_epoch_MLP.png">

* Loss function vs epochs in CNN model

**Confusion matrix**
* Confusion matric in MLP model
<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Confusion_Matrix_MLP.png">

* Confusion matric in CNN model
<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/Los_vs_epoch_MLP.png">

**Classification report**
* Classification report for MLP model
<img src="https://github.com/avtnguyen/image-classification-mlp-cnn/blob/main/image/classification_report_MLP.png">

**2. Feature selection:** I performed feature selection based on the univariate statistical tests by computing the ANOVA F-value betwen the numerical features (e.g., f_1, f_2...) and the label target. The new dataset includes the most 25 features and f_46 because it is a categorical feature. 

**3. Splitting the dataset** to train test sets based on the following specifications: Train size: 75%, test size: 25%, stratifying based on the y label  to ensure that both the train and test sets have the same class proportion similar to the original dataset. After that, I normalized both train and test datasets using the StandardScaller() to remove the mean and scaling to unit variance. 

**4. Data augmentation**: Since the dataset is highly imbalanced, i implemented multiple data augmentation techniques to improve the quality of the dataset based on the following algorithms:

* Synthetic Minority Oversampling Technique(SMOTE): The sample in minority class is first selected randomly and its k nearest minority class neighbors are found based on the K-nearest neighbors algorithm. The synthetic data is generated between two instances in feature space. 
* Adaptive Synthetic Sampling (ADASYN): The synthetic data for minority class is generated based on the desnity distribution of the minority class. Specifically, more data is created in area with low density of minority class and less data is generated in area with high density of minority example
* SMOTE-TOMEK: Combine SMOTE and TOMEK techniqes where the oversampling technique for minority class and the cleaning using Tomek links.  
* SMOTE- ENN: Combine SMOTE and Edited Nearest Neighbours (ENN) techniques where the oversampling technique for minority class and the cleaning using ENN

Source: [Imbalanced learn](https://imbalanced-learn.org/stable/references/over_sampling.html)
 
**5. Build a simple deep learning network** and combine with multiple data augmentation techniques [See code here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill_detection_deepLearningModel.ipynb)
<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/DNN_summary.png">

**6. Implement ensemble learning algorithms**, which include Random Forest, and XGBoost, and compare the model performance given the unbalanced dataset for oil spill detection [See code here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill_detection_model.ipynb)

### Evaluation metrics
For imbalance dataset and classification model, the following metrics are used to evaluate the model performance:
* Precision
* Recall
* f1 score

### Results:
- Both the precision, recall and f1 scores are low in all models. This could be due to the small imbalance dataset that we have.
- Given the dataset without any resampling technique, XGBoost outperformed other algorithms.
- When data augmentation technique is implemented, the performance of Random Forrest model is improved significantly using SMOTE+TOMEK technique as shown in table below.
- More data is needed to improve the model accuracy for oil spill detection


| model       | resample                     | precision  | recall | f1   |
| ------------|:----------------------------:| ----------:|-------:|-----:|
| DNN         | SMOTE                        |   0.267    |0.8     |0.4   |
| DNN         | SMOTE+TOMEK                  |   0.385    |0.5     |0.435 |
| RF          | SMOTE+TOMEK                  |  0.625     |0.5     |0.555 |
| RF          | SMOTE+ENN                    |   0.461    |0.6     |0.522 |
| XGBoost     | ADASYN                       |   0.5      |0.5     |0.5   |
| XGBoost     | SMOTE                        |   0.454    |0.5     |0.476 |
| DNN         | No resample                  |   0.114    |0.9     |0.2   |
| RF          | No resample                  |   0.2      |0.1     |0.133 |
| XGBoost     | No resample                  |   0.357    |0.5     |0.417 |



### Methods Used
* Data Cleaning and Wrangling
* Data Analysis
* Data Visualization
* Data Augmentation
* Machine Learning Model: Deep learning network, Random Forest, XGBoost

### Technologies
* Pandas
* Numpy
* Seaborn and Pyplot
* sklearn
* TensorFlow
* imbalanced-learn
* Colab

## Needs of this project
- Data exploration/descriptive statistics
- Data processing/cleaning
- Data modeling
- Writeup/reporting

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is can be dowloaded from [this repository](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill.csv) or from [Kaggle](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)
3. Data processing and modeling scripts are being kept [here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/)

## References:
* https://deepai.org/machine-learning-glossary-and-terms/feed-forward-neural-network

## Contributing Members

**Team Leads (Contacts) : [Anh Nguyen ](https://github.com/avtnguyen)**

## Contact
* Feel free to contact team leads with any questions or if you are interested in contributing!
