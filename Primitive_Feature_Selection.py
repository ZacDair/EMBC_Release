import numpy as np
import os, sys

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from Modules import utility

arg = sys.argv

# Load data file
datasetFile = 'Data/dataset_smile_challenge.npy'
dataset = np.load(datasetFile, allow_pickle=True).item()
print(f"Loading data from {datasetFile}")

# Train/Test Split
dataset_train = dataset['train']
dataset_test = dataset['test']

# print(f"Original Train Shape {dataset_train.shape}, Original Test Shape {dataset_test.shape}")

# Get features dict and labels
trainFeatures = utility.getFeatures(dataset_train)
trainLabels = utility.binarizeLabels(dataset_train['labels'])

testFeatures = utility.getFeatures(dataset_test)
testLabels = utility.binarizeLabels(dataset_test['labels'])

print("Features loaded into dictionary object\nLabels converted to binary values")

def perFeatureClassifier():
    # Returns list of {'name':'logReg', 'model':LogisticRegression()}
    models = utility.sklearnModelList()

    # Loop through models
    for m in models:
        clf = m['model']
        name = m['name']

        for feature in trainFeatures.keys():
            if feature == "Deep_Masking" or feature == "ECG_Masking" or feature == "GSR_Masking":
                pass
            else:
                # Subset of trainData for training
                featureData = utility.convertFeatures(trainFeatures[feature])
                clf.fit(featureData[0:1700], trainLabels[0:1700])

                # Test on remaining trainData
                predictions = clf.predict(featureData[1701:2700])

                try:
                    score = accuracy_score(trainLabels[1701:2070], predictions)
                    f1 = f1_score(trainLabels[1701:2070], predictions)
                    conf = confusion_matrix(trainLabels[1701:2070], predictions)
                    sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
                    spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])
                except ValueError:
                    avgResult = np.average(predictions.reshape(-1, 1), axis=1)
                    preds_edit = (avgResult > 0.5).astype(int)
                    score = accuracy_score(trainLabels[1701:2070], preds_edit)
                    f1 = f1_score(trainLabels[1701:2070], preds_edit)
                    conf = confusion_matrix(trainLabels[1701:2070], preds_edit)
                    sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
                    spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])

                print(f"Feature: {feature} Accuracy: {score}, F1: {f1}, Sensitivity: {sens}, Specificity: {spec}\n")


def manualCombinedFeatures():
    # Returns list of {'name':'logReg', 'model':LogisticRegression()}
    models = utility.sklearnModelList()

    # Loop through models
    for m in models:
        clf = m['model']
        name = m['name']

        featureData = utility.convertFeatures(trainFeatures['ECG_features'])
        featureData_1 = utility.convertFeatures(trainFeatures['GSR_features'])
        featureData_2 = utility.convertFeatures(trainFeatures['ECG_features_T'])
        featureData_3 = utility.convertFeatures(trainFeatures['ECG_features_C'])
        res = np.concatenate((featureData, featureData_1, featureData_2, featureData_3), axis=1)

        # Fit to subset of training data
        clf.fit(res[0:1700], trainLabels[0:1700])

        # Test on remaining trainData
        predictions = clf.predict(res[1701:2700])

        # Attempt computation of evaluation metrics - trigger exception on regression
        try:
            score = accuracy_score(trainLabels[1701:2070], predictions)
            f1 = f1_score(trainLabels[1701:2070], predictions)
            conf = confusion_matrix(trainLabels[1701:2070], predictions)
            sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
            spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])
        except ValueError:
            avgResult = np.average(predictions.reshape(-1, 1), axis=1)
            preds_edit = (avgResult > 0.5).astype(int)
            score = accuracy_score(trainLabels[1701:2070], preds_edit)
            f1 = f1_score(trainLabels[1701:2070], preds_edit)
            conf = confusion_matrix(trainLabels[1701:2070], preds_edit)
            sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
            spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])

        print(f"{name},{'All Features'}, Acc {score},F1 {f1},Sens {sens},Spec {spec}")


# Ensemble
def individualFeatureEnsemble():
    models = utility.sklearnModelList()
    scores = []
    preds = []
    trainedModels = {}

    # Classifier type
    m = models[3]  # Extra Trees
    for feature in trainFeatures.keys():
        if feature == "Deep_Masking" or feature == "ECG_Masking" or feature == "GSR_Masking":
            pass
        else:
            clf = m['model']
            featureData = utility.convertFeatures(trainFeatures[feature])
            print(feature)
            print(featureData.shape)
            trainedModels[feature] = clf.fit(featureData[0:1070], trainLabels[0:1070])

            predictions = clf.predict(featureData[1071:2070])
            score = clf.score(featureData[1071:2070], trainLabels[1071:2070])
            scores.append(score)
            preds.append(predictions)


    preds = sum(preds)
    preds = utility.binarizeLabels(preds)

    # correct = 0
    # for estimate, actual in zip(preds, trainLabels[1701:2070]):
    #     if estimate == actual:
    #         correct += 1

    try:
        score = accuracy_score(trainLabels[1071:2070], preds)
        f1 = f1_score(trainLabels[1071:2070], preds)
        conf = confusion_matrix(trainLabels[1071:2070], preds)
        sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
        spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])
    except ValueError:
        avgResult = np.average(preds.reshape(-1, 1), axis=1)
        preds_edit = (avgResult > 0.5).astype(int)
        score = accuracy_score(trainLabels[1071:2070], preds_edit)
        f1 = f1_score(trainLabels[1071:2070], preds_edit)
        conf = confusion_matrix(trainLabels[1071:2070], preds_edit)
        sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
        spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])

    # print(f"Feature: {feature} Accuracy: {score}, F1: {f1}, Sensitivity: {sens}, Specificity: {spec}\n")
    print(f"Ensemble per Feature,all,Acc {score},F1 {f1},Sens {sens},Spec {spec}")


# Run individual experiments
print("Per Feature Classification - Feature Selection")
perFeatureClassifier()
print("Combined Feature Ensemble")
individualFeatureEnsemble()
print("Feature Permutatons - Manual Combination of features (default=all)")
manualCombinedFeatures()

''' Train the full model and attempt testing predictions '''
print("Training on the full train set - RandomForest")
featureData = utility.convertFeatures(trainFeatures['ECG_features'])
featureData_1 = utility.convertFeatures(trainFeatures['GSR_features'])
featureData_2 = utility.convertFeatures(trainFeatures['ECG_features_T'])
featureData_3 = utility.convertFeatures(trainFeatures['ECG_features_C'])
res = np.concatenate((featureData, featureData_1, featureData_2, featureData_3), axis=1)

clf = RandomForestClassifier()
clf.fit(res, trainLabels)
# Test on remaining trainData
# predictions = clf.predict(trainFeatures[feature][1701:2700])
featureData = utility.convertFeatures(testFeatures['ECG_features'])
featureData_1 = utility.convertFeatures(testFeatures['GSR_features'])
featureData_2 = utility.convertFeatures(testFeatures['ECG_features_T'])
res = np.concatenate((featureData, featureData_1, featureData_2), axis=1)
predictions = clf.predict(res)

utility.uniqueValueCount(predictions)
utility.writePredictionsToFile(predictions, "Results/all_features_random_forest_output.txt")