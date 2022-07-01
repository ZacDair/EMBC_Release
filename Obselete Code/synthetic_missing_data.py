import numpy as np
import pickle
import pandas as pd
from Modules import utility

# Load data file
datasetFile = 'Data/dataset_smile_challenge.npy'
dataset = np.load(datasetFile, allow_pickle=True).item()
print(f"Loading data from {datasetFile}")

# Train/Test Split
dataset_train = dataset['train']
dataset_test = dataset['test']


# Get features dict and labels
trainFeatures = utility.getFeatures(dataset_train)
trainLabels = utility.binarizeLabels(dataset_train['labels'])

# trainFeatures = utility.getFeatures(dataset_test)
# trainLabels = utility.binarizeLabels(dataset_test['labels'])

testFeatures = utility.getFeatures(dataset_test)
testLabels = utility.binarizeLabels(dataset_test['labels'])

ecg_feature_neutral = []
gsr_feature_neutral = []
deep_ecg_c_neutral = []
deep_ecg_t_neutral = []

ecg_feature_stress = []
gsr_feature_stress = []
deep_ecg_c_stress = []
deep_ecg_t_stress = []

print("Features loaded into dictionary object\nLabels converted to binary values")

# mean_hr, std_rr, rms_sdrr, low_freq, high_freq, low_high_freq_ratio, very_low_ratio, hr_cycle
index = 0
for ecg_feature, gsr_feature,deep_ecg_c, deep_ecg_t, deep_mask, ecg_mask, gsr_mask, labels in zip(trainFeatures["ECG_features"], trainFeatures["GSR_features"], trainFeatures["ECG_features_C"], trainFeatures["ECG_features_T"], trainFeatures["Deep_Masking"], trainFeatures["ECG_Masking"], trainFeatures["GSR_Masking"], trainLabels):
    min = 0

    while min < 60:

        if deep_mask[min] == 0.0:
            deepMissing = True
        else:
            if labels == 0.0:
                deep_ecg_c_neutral.append(deep_ecg_c[min])
                deep_ecg_t_neutral.append(deep_ecg_t[min])
            else:
                deep_ecg_c_stress.append(deep_ecg_c[min])
                deep_ecg_t_stress.append(deep_ecg_t[min])

        if ecg_mask[min] == 0.0:
            ecgMissing = True
        else:
            if labels == 0.0:
                ecg_feature_neutral.append(ecg_feature[min])
            else:
                ecg_feature_stress.append(ecg_feature[min])

        if gsr_mask[min] == 0.0:
            gsrMissing = True
        else:
            if labels == 0.0:
                gsr_feature_neutral.append(gsr_feature[min])
            else:
                gsr_feature_stress.append(gsr_feature[min])

        min += 1
    index += 1

# Get the column average of each list

avg_ecg_feature_neutral = np.nanmean(ecg_feature_neutral, axis=0)
avg_gsr_feature_neutral = np.nanmean(gsr_feature_neutral, axis=0)
avg_deep_ecg_c_neutral = np.nanmean(deep_ecg_c_neutral, axis=0)
avg_deep_ecg_t_neutral = np.nanmean(deep_ecg_t_neutral, axis=0)

avg_ecg_feature_stress = np.nanmean(ecg_feature_stress, axis=0)
avg_gsr_feature_stress = np.nanmean(gsr_feature_stress, axis=0)
avg_deep_ecg_c_stress = np.nanmean(deep_ecg_c_stress, axis=0)
avg_deep_ecg_t_stress = np.nanmean(deep_ecg_t_stress, axis=0)

# print("e_n",avg_ecg_feature_neutral)
# print("g_n",avg_gsr_feature_neutral)
# print("dc_n",avg_deep_ecg_c_neutral)
# print("dt_n",avg_deep_ecg_t_neutral)
#
# print(avg_ecg_feature_stress)
# print(avg_gsr_feature_stress)
# print(avg_deep_ecg_c_stress)
# print(avg_deep_ecg_t_stress)

index = 0
for ecg_feature, gsr_feature,deep_ecg_c, deep_ecg_t, deep_mask, ecg_mask, gsr_mask, labels in zip(trainFeatures["ECG_features"], trainFeatures["GSR_features"], trainFeatures["ECG_features_C"], trainFeatures["ECG_features_T"], trainFeatures["Deep_Masking"], trainFeatures["ECG_Masking"], trainFeatures["GSR_Masking"], trainLabels):
    min = 0

    while min < 60:

        if deep_mask[min] == 0.0:
            if labels == 0.0:
                trainFeatures['ECG_features_C'][index][min] = avg_deep_ecg_c_neutral
                trainFeatures['ECG_features_T'][index][min] = avg_deep_ecg_t_neutral
            else:
                trainFeatures['ECG_features_C'][index][min] = avg_deep_ecg_c_stress
                trainFeatures['ECG_features_T'][index][min] = avg_deep_ecg_t_stress

        if ecg_mask[min] == 0.0:
            if labels == 0.0:
                trainFeatures["ECG_features"][index][min] = avg_ecg_feature_neutral
            else:
                trainFeatures["ECG_features"][index][min] = avg_ecg_feature_stress

        if gsr_mask[min] == 0.0:
            if labels == 0.0:
                trainFeatures["GSR_features"][index][min] = avg_gsr_feature_neutral
            else:
                trainFeatures["GSR_features"][index][min] = avg_gsr_feature_stress

        min += 1
    index += 1

featureDict = trainFeatures
sampleCount = 2070
labels = trainLabels
name = "synthetic_missing_data_training"

mainDf = pd.DataFrame()
for feature in featureDict:
    print(featureDict[feature].shape)

    if featureDict[feature].ndim == 3:
        tempDf = pd.DataFrame(featureDict[feature].reshape(sampleCount*60, -1))
    else:
        tempDf = pd.DataFrame(featureDict[feature].reshape(sampleCount*60, 1))

    # tempDf[feature] = tempDf.values.tolist()
    # mainDf[feature] = tempDf[feature]
    mainDf = pd.concat([mainDf, tempDf], ignore_index=True, axis=1)


# Add labels
tempDf = pd.DataFrame(labels.repeat(repeats=60))
mainDf = pd.concat([mainDf, tempDf], ignore_index=True, axis=1)
# mainDf['label'] = tempDf.values.tolist()

mainDf.to_pickle("Data/"+name+".pkl")
mainDf.to_csv("Data/"+name+".csv", sep=",")

