# Image Classification with MLP and CNN

# Oil-Spill-Detection-Model

#### -- Project Status: [Completed]

## Project Objective
The purpose of this project is to build a machine learning model to detect oil spill from satellite image using the processed dataset provided from Kaggle. In this project, i look at multiple different data augmentation techniques (SMOTE, ADASYN, etc) for imbalanced dataset and different supervised learning algorithms to improve the model performance. 

## Project Description
**About the dataset** The dataset was developed by starting with satellite images of the ocean, some of which contain an oil spill and some that do not. Images were split into sections and processed using computer vision algorithms to provide a vector of features to describe the contents of the image section or patch.
The task is, given a vector that describes the contents of a patch of a satellite image, then predicts whether the patch contains an oil spill or not, e.g. from the illegal or accidental dumping of oil in the ocean. There are two classes and the goal is to distinguish between spill and non-spill using the features for a given ocean patch. 

Non-Spill: negative case, or majority class.
Oil Spill: positive case, or minority class.
[Source](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)

### Project Pipeline :
**1. Data processing and exploration:** In this section, I performed data cleaning to remove NaN vaues and features that contains only zero values. I also checked for imblance classes and the data distribution as shown in the table below

| Category       | Total values  | Percentage(%)  |
| -------------  |:-------------:| --------------:|
| Oil spill      | 41            | 4.38           |
| No oil spill   | 896           |  95.62         |

<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/Label_distribution.png">

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
* https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection
* https://imbalanced-learn.org/stable/references
* https://machinelearningmastery.com/framework-for-imbalanced-classification-projects

## Contributing Members

**Team Leads (Contacts) : [Anh Nguyen ](https://github.com/avtnguyen)**

## Contact
* Feel free to contact team leads with any questions or if you are interested in contributing!
