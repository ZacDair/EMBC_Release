import numpy as np
import pickle
import pandas as pd
from Modules import utility
import os.path

datasetFile = 'Data/dataset_smile_challenge.npy'

if not os.path.exists(datasetFile):
    print(f"ERROR: it seems like the following file is missing:\n {datasetFile}")
    exit()

# Load data file
dataset = np.load(datasetFile, allow_pickle=True).item()
print(f"Loading data from {datasetFile}")

# Train/Test Split
dataset_train = dataset['train']
dataset_test = dataset['test']


# Get features dict and labels
name = "synthetic_missing_data_training_avg"
sampleCount = 2070
trainFeatures = utility.getFeatures(dataset_train)
trainLabels = utility.binarizeLabels(dataset_train['labels'])

''' Uncomment to run average feature imputation on testing data'''
# name = "synthetic_missing_data_testing_avg"
# sampleCount = 986
# trainFeatures = utility.getFeatures(dataset_test)
# trainLabels = utility.binarizeLabels(dataset_test['labels'])

ecg_feature_list = []
gsr_feature_list = []
deep_ecg_c_list = []
deep_ecg_t_list = []

print("Features loaded into dictionary object\nLabels converted to binary values")
print("\nComputing Average Feature Value...")
index = 0
for ecg_feature, gsr_feature,deep_ecg_c, deep_ecg_t, deep_mask, ecg_mask, gsr_mask, labels in zip(trainFeatures["ECG_features"], trainFeatures["GSR_features"], trainFeatures["ECG_features_C"], trainFeatures["ECG_features_T"], trainFeatures["Deep_Masking"], trainFeatures["ECG_Masking"], trainFeatures["GSR_Masking"], trainLabels):
    min = 0

    while min < 60:

        if deep_mask[min] == 0.0:
            deepMissing = True
        else:
            deep_ecg_c_list.append(deep_ecg_c[min])
            deep_ecg_t_list.append(deep_ecg_t[min])

        if ecg_mask[min] == 0.0:
            ecgMissing = True
        else:

            ecg_feature_list.append(ecg_feature[min])

        if gsr_mask[min] == 0.0:
            gsrMissing = True
        else:
            gsr_feature_list.append(gsr_feature[min])

        min += 1
    index += 1

# Get the column average of each list
avg_ecg_feature = np.nanmean(ecg_feature_list, axis=0)
avg_gsr_feature = np.nanmean(gsr_feature_list, axis=0)
avg_deep_ecg_c = np.nanmean(deep_ecg_c_list, axis=0)
avg_deep_ecg_t = np.nanmean(deep_ecg_t_list, axis=0)

print("\nReplace Missing Values With The Average Feature Value...")
index = 0
for ecg_feature, gsr_feature,deep_ecg_c, deep_ecg_t, deep_mask, ecg_mask, gsr_mask, labels in zip(trainFeatures["ECG_features"], trainFeatures["GSR_features"], trainFeatures["ECG_features_C"], trainFeatures["ECG_features_T"], trainFeatures["Deep_Masking"], trainFeatures["ECG_Masking"], trainFeatures["GSR_Masking"], trainLabels):
    min = 0

    while min < 60:

        if deep_mask[min] == 0.0:
            trainFeatures['ECG_features_C'][index][min] = avg_deep_ecg_c
            trainFeatures['ECG_features_T'][index][min] = avg_deep_ecg_t

        if ecg_mask[min] == 0.0:
            trainFeatures["ECG_features"][index][min] = avg_ecg_feature

        if gsr_mask[min] == 0.0:
            trainFeatures["GSR_features"][index][min] = avg_gsr_feature

        min += 1
    index += 1

featureDict = trainFeatures
labels = trainLabels


# Output to a single DataFrame
mainDf = pd.DataFrame()
for feature in featureDict:

    # Reshape hourly into per minute features
    if featureDict[feature].ndim == 3:
        tempDf = pd.DataFrame(featureDict[feature].reshape(sampleCount*60, -1))
    else:
        tempDf = pd.DataFrame(featureDict[feature].reshape(sampleCount*60, 1))

    mainDf = pd.concat([mainDf, tempDf], ignore_index=True, axis=1)


# Add labels
tempDf = pd.DataFrame(labels.repeat(repeats=60))
mainDf = pd.concat([mainDf, tempDf], ignore_index=True, axis=1)

# Output as CSV and PKL
mainDf.to_pickle("Data/"+name+".pkl")
mainDf.to_csv("Data/"+name+".csv", sep=",")

