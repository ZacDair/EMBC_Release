import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from Modules import utility
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import os.path

trainingFile = "Data/SMILE_Training_DataFrame.pkl"
testingFile = "Data/SMILE_Training_DataFrame.pkl"

if not os.path.exists(trainingFile) or not os.path.exists(testingFile):
    print(f"ERROR: it seems like the following files are missing:\n {trainingFile}\n {testingFile}")
    exit()

traindf = pd.read_pickle(trainingFile)
testdf = pd.read_pickle(testingFile)

# create a scaler object
# scaler = MinMaxScaler()
# fit and transform the data

# Relabel according to origin split (1=Train, 0=Test)
traindf.iloc[:, 343] = 1
testdf.iloc[:, 343] = 0

# traindf = pd.DataFrame(scaler.fit_transform(traindf), columns=traindf.columns)
# testdf = pd.DataFrame(scaler.fit_transform(testdf), columns=testdf.columns)

# Combine both DataFrames
combined = pd.concat([traindf, testdf])

# combined_scaled = pd.DataFrame(scaler.fit_transform(combined), columns=combined.columns)
# combined = combined_scaled
# combined.iloc[0:len(traindf), 343] = 1
# combined.iloc[len(traindf)+1:len(traindf)+len(testdf), 343] = 0

# Shuffle the dataset
combined = combined.sample(frac=1).reset_index(drop=True)

# Replace any np.inf, -np.inf, np.nan values
combined.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

# Get 80% mark of the data - use that to create a train and pre-submission testing set
upperBound = len(combined)
upperBound = int((80/100) * upperBound)

training = combined[0:upperBound-1]
features = training.iloc[:, 320:324]
trainLabels = training.iloc[:, -1]
training.drop(training.columns[-1], axis=1, inplace=True)

testing = combined[upperBound:len(combined)]
testfeatures = testing.iloc[:, 320:324]
testLabels = testing.iloc[:, -1]
testing.drop(testing.columns[-1], axis=1, inplace=True)

models = utility.sklearnModelList()
bestModel = ""
bestScore = -1
print("Running baseline models - Detecting Covariate Shift...")
for m in models:
    clf = m['model']
    name = m['name']

    # Subset of trainData for training
    clf.fit(features, trainLabels)

    # Test on remaining trainData
    preds = clf.predict(testfeatures)

    try:
        score = accuracy_score(testLabels, preds)
        f1 = f1_score(testLabels, preds)
        conf = confusion_matrix(testLabels, preds)
        sens = conf[0,0]/(conf[0,0]+conf[0,1])
        spec = conf[1,1]/(conf[1,0]+conf[1,1])
    except ValueError:
        avgResult = np.average(preds.reshape(-1, 1), axis=1)
        preds_edit = (avgResult > 0.5).astype(int)
        score = accuracy_score(testLabels, preds_edit)
        f1 = f1_score(testLabels, preds_edit)
        conf = confusion_matrix(testLabels, preds_edit)
        sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
        spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])

    if score > bestScore:
        bestScore = score
        bestModel = m["model"]
        bestModelObj = clf
        bestPreds = preds

    print(f"{name},{score},{f1},{sens},{spec}")