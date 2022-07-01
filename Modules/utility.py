import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Binarize labels with threshold (l >= 1) == 1, (l < 1) == 0
def binarizeLabels(labels):
    indices = labels >= 1
    labels[indices] = 1
    # Clean any np.NAN, np.infinite
    return np.nan_to_num(labels)


# Convert to hour long features
def convertFeatures(featureData):
    if featureData.ndim > 2:
        sampleCount, rowCount, featureCount = featureData.shape
        featureData = featureData.reshape(sampleCount, featureCount*rowCount)
    # Clean any np.NAN, np.infinite
    return np.nan_to_num(featureData)


# Print unique value counts
def uniqueValueCount(data):
    unique, counts = np.unique(data, return_counts=True)
    print(np.asarray((unique, counts)).T)


# Retrieve features and labels
def getFeatures(data):
    # Deep Features
    deep_features = data['deep_features']
    # conv1d backbone based features for ECG signal.
    ECG_C = deep_features['ECG_features_C']
    # transformer backbone basde features for ECG signal
    ECG_T = deep_features['ECG_features_T']

    # Handcrafted Features
    handcrafted_features = data['hand_crafted_features']
    # handcrafted features for ECG signal
    ECG_HC = handcrafted_features['ECG_features']
    # handcrafted features for GSR signal.
    GSR_HC = handcrafted_features['GSR_features']

    # Deep_ECG Masking - (0=missing, 1=present) of deep features for ECG
    MASK_ECG_DEEP = deep_features['masking']

    # HC_ECG Masking - (0=missing, 1=present) of handcrafted features for ECG
    MASK_ECG_HC = handcrafted_features['ECG_masking']

    # HC_GSR Masking - (0=missing, 1=present) of handcrafted features for GSR
    MASK_GSR_HC = handcrafted_features['GSR_masking']

    return {"ECG_features_C": ECG_C, "ECG_features_T": ECG_T, "ECG_features": ECG_HC, "GSR_features": GSR_HC, "Deep_Masking":MASK_ECG_DEEP, "ECG_Masking":MASK_ECG_HC, "GSR_Masking":MASK_GSR_HC}
    #return {"ECG_features_C": ECG_C, "ECG_features_T": ECG_T, "ECG_features": ECG_HC, "GSR_features": GSR_HC}


# Bunch of models listed in dict form including names
def sklearnModelList():
    models = [
        {'name': 'LogReg', 'model': LogisticRegression(max_iter=2500)},
        {'name': 'Perceptron', 'model': Perceptron()},
        {'name': 'RandForest', 'model': RandomForestClassifier()},
        {'name': 'SVC-Lin', 'model': SVC(kernel='linear')},
        {'name': 'SVC-sig', 'model': SVC(kernel='sigmoid')},
        {'name': 'SVC-rbf', 'model': SVC(kernel='rbf')},
        {'name': 'SVC-poly', 'model': SVC(kernel='poly')},
        {'name': 'ExtraTrees', 'model': ExtraTreesClassifier()},
        {'name': 'AdaBoost', 'model': AdaBoostClassifier()},
        {'name': 'XGBoost', 'model': XGBRegressor()},
        {'name': 'LDA', 'model': LinearDiscriminantAnalysis()},
        {'name': 'LinearReg', 'model': LinearRegression()},
        {'name': 'GaussianNB', 'model': GaussianNB()},
        {'name': 'DecisionTreeClassifier', 'model': DecisionTreeClassifier()},
        {'name': 'KNN', 'model': KNeighborsClassifier()}
    ]
    return models


# Output the predictions given in the expected format, one per line
def writePredictionsToFile(predictions, outputFile):
    outputFile = os.path.join("Results", outputFile)
    with open(outputFile, "w+") as f:
        for line in predictions:
            f.write(str(int(line)))
            f.write("\n")
        f.close()


# Convert dataset to pandas dataframe
def convertToDataFrame(data, name, sampleCount):
    # index, ecg_f_c, ecg_f_t, ecg_d_masking, hc_ecg, hc__ecg_masking, hc_gsr, hc_gsr_masking, label
    columns = ["ECG_features_C", "ECG_features_T", "Deep_masking", "ECG_features", "GSR_features", "ECG_masking", "GSR_masking", "label"]
    mainDf = pd.DataFrame()

    featureDict = getFeatures(data)
    labels = binarizeLabels(data['labels'])

    for feature in featureDict:

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
    return mainDf


# Feature Correlation Table
def featureCorrelation(df):
    corr = df.corr()
    plt.subplots(figsize=(20, 15))
    sns.heatmap(corr, annot=True)
    plt.show()