# EMBC - Stress Detection Challenge

## The resulting publication can be found on Arxiv  
"Classification of Stress via Ambulatory ECG and GSR Data",  
Zachary Dair, Muhammad Muneeb Saad, Urja Pawar, Samantha Dockray, Ruairi O'Reilly  
[Arxiv preprint available](https://arxiv.org/abs/2208.04705)  
Awaiting inclusion in the book associated with this challenge organised by the Compwell Group.  

## Getting Started  
**Steps to run the optimal approach:**
 1. download the SMILE dataset
 2. Place the .npy file in the Data folder
 3. Run Main_Classification.py (A train and test pandas DataFrame will be created)
 4. Required preprocessing and shuffling will automatically take place
 5. Results will appear in console and in Logs

The goal of this workshop: is to inspire ideas and collaborations, raise awareness of reproducibility problems in
modeling wearable data in the wild, and drive the research frontier.
Publicly sharing the datasets, including both features and baselines, will accelerate research activity
such as multimodal wearable/mobile sensor data processing and modeling, handling missing data, and personalization.

**Accepted papers fit under:**
1. Research papers about the detection/prediction of stress and mental health using wearable/mobile sensors
2. Research papers about technical solutions using our open stress datasets
3. Challenge papers in which authors described a specific challenge to be pitched and discussed at the workshop. 

## Dataset
For more details see [here](https://compwell.rice.edu/workshops/embc2022/dataset) and [here](https://compwell.rice.edu/workshops/embc2022/challenge/)

## Feature Imputation Data Files
If the feature imputation data files are required - please contact [zachary.dair@mycit.ie]()    

## Features
# Table of Handcrafted Features
![Table of handcrafted features from EMbC Compwell website](https://lh3.googleusercontent.com/cKf_vUPEWAf4j7laC293Bsr3BZchO0L-yh0hZ-xD5ti1nJc9ZXE6yXug41m04Zq6MFpr0sDyCX4n69m65xmJcfbjcus06TioLaDGlI_Qlk5VV5aOaUqBpZJG8mZRwfuMhQ=w1280)

# Deep Features
Deep features are extracted through unsupervised machine learning on the whole SMILE dataset.
Two forms of deep features are available for ECG - extracted by Transformer and extracted by Conv-1D.
[More info here](https://compwell.rice.edu/workshops/embc2022/dataset)

# Masking
Masking values are present in the deep features section - indicating the presence (1) or absence (0) of deep features for that period
Similarly these are also present for ECG and GSR in handcrafted features under the same premise.

## Current Code

# Directory Structure
```
|---README.md
|---.gitignore
|---Covariate_Shift_Comparison.py
|---Feature_Imputation_Avg.py
|---Feature_Imputation_Euclidiean.py
|---Primitive_Feature_Selection.py
|---Main_Classification.py (Entry Point)
|
|---Data
|   \-- ...
|
|---Modules
|   |-- __init__.py
|   \-- utility.py
|
\----Results
     \-- Template

```

# Data
This is the expected location for the "dataset_smile_challenge.npy" file.  
Additionally any extra data files created as part of experimental runs will go here.  
For example: SMILE_Training_DataFrame.csv(.pkl), SMILE_Testing_DataFrame.csv(.pkl), and files resulting from Feature Imputation.  

# Results
any_occurence.txt - Occurence based voting mechanism result (combining 60 predictions into a single hour result)  
basic_point1_threshold.txt - Mean label threshold based voting mechanism result (combining 60 predictions into a single hour result)  
LDA_Voter.txt - LDA classifier based voting mechanism result (combining 60 predictions into a single hour result)   
answer_template.txt is the template for how to present predictions to be evaulated.
One predictions per line (1 or 0 values) - created using utility.py - writePredictionsToFile(predictions, outputFile)

# Code
Main_Classification.py - Core code - preset to run and provide optimal approach result - requires manually editing to change feature indexes and file loading to use the results of feature imputation   
**NOTE:** As the labels given for testing are not the true labels, a subset of the dataset was used for model evaluation    
Covariate_Shift.py - Converts the labels of the train/test data into (1=train, 0:Test), combines data splits and shuffles and aims to classify the origin split (Train or Test), high accuracy indicates there is a covariate shift.
Feature_Imputation_Avg.py - uses average feature values to replace missing values - requires two seperate runs one for training and one for testing - uncommenting code at the top
Feature_Imputation_Euclidiean.py - uses Euclidiean distance to find similar feature values to replace missing values - requires two seperate runs one for training and one for testing - uncommenting code at the top
Primitive_Feature_Selection.py - handles individual feature, combined feature, and ensemble of individual feature classifiers to provide feature selection insights. 

Modules - utility.py - contains several helper functions:
- including converting stress labels to binary (1 or 0)
- Dictionary of basic models and names
- Writing predictions to a file
- Getting all features from a dataset split - including maskings (not returned but can be added to the returned dict)
- Unique value count from numpy array - similar to pandas.DataFrame.value_counts()
- Converting features from a numpy array (x, 60, 8) into (x, 60*8) - waiting on response to ensure this is valid, premise seems to be that we have 2070 data samples, which each have 60 arrays containing the features.
