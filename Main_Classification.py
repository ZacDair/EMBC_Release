import pandas as pd
import numpy as np
from Modules import utility
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, plot_roc_curve, plot_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import shap
import os.path


# Load data file - and convert to a per minute DataFrame
if not os.path.exists("SMILE_Training_DataFrame.pkl") or not os.path.exists("SMILE_Testing_DataFrame.pkl"):
    datasetFile = 'Data/dataset_smile_challenge.npy'
    dataset = np.load(datasetFile, allow_pickle=True).item()
    print(f"Loading data from {datasetFile}")
    print("Converting into DataFrames...")
    # Train/Test Split
    dataset_train = dataset['train']
    dataset_test = dataset['test']
    trainingDf = utility.convertToDataFrame(dataset_train, "SMILE_Training_DataFrame", 2070)
    testingDf = utility.convertToDataFrame(dataset_test, "SMILE_Testing_DataFrame", 987)

# DataFrame Structure
# Col 0 index
# Col 1-255 Deep_ECG_C
# Col 256-320 Deep_ECG_T
# Col 321-328 ECG
# Col 329-340 GSR
# Masking 341
# ECG Masking 342
# GSR Masking 343
# Col 344 label


# Load our training and testing data using DataFrames
print("Attempting to load DataFrames...")
trainingDf = pd.read_pickle("Data/SMILE_Training_DataFrame.pkl")
testingDf = pd.read_pickle("Data/SMILE_Testing_DataFrame.pkl")
shuffle = True
print("Loaded DataFrames...")

# Get 80% mark of the data - use that to create a train and pre-submission testing set
print("Getting sub-train/test split...")
upperBound = len(trainingDf)
upperBound = int((80/100) * upperBound)

# Shuffle the DataFrame
if shuffle:
    print("Shuffling Training Data...")
    trainingDf = trainingDf.sample(frac=1).reset_index(drop=True)

# Replace any np.inf, -np.inf, np.nan values
trainingDf.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

# Get a subset of our training data
training = trainingDf[0:upperBound-1]
trainLabels = training.iloc[:, -1]

# Define which features to utilise - 320:341 (ECG, GSR)
features = training.iloc[:, 320:341]

# Remove the label column
training.drop(training.columns[-1], axis=1, inplace=True)

# Get a subset of our testing data - for development evaluation
testing = trainingDf[upperBound:len(trainingDf)]

# Define which features to utilise - 320:341 (ECG, GSR)
testFeatures = testing.iloc[:, 320:341]
testLabels = testing.iloc[:, -1]

# Remove the label column
testing.drop(testing.columns[-1], axis=1, inplace=True)


# Run model selection - logging results
print("Running classification against baseline models...")
currentTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
with open("Logs/"+currentTime+".txt", "w+") as f:

    # Write header
    head = f"Model, Accuracy, F1_Score, Sensitivity, Specificity, trainCount, testCount\n"
    f.write(head)

    models = utility.sklearnModelList()
    bestModel = ""
    bestModelObj = ""
    bestScore = -1
    bestPreds = 0
    for m in models:
        # print(f"Time started: {datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        clf = m['model']
        name = m['name']

        # Subset of trainData for training
        clf.fit(features, trainLabels)

        # Test on remaining trainData
        preds = clf.predict(testFeatures)

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

        output = f"{name}, {score}, {f1}, {sens}, {spec}, {len(training)}, {len(preds)}\n"
        f.write(output)
        print(f"Model: {name}, Features: {'ECG+GSR'}, Accuracy: {score}, F1-Score: {f1},Sensitivity: {sens},Specificity: {spec}")


sns.set_theme()
plot_roc_curve(bestModelObj, testFeatures, testLabels)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Optional feature importance - very slow...
# explainer = shap.TreeExplainer(bestModelObj)
# shap_values = explainer.shap_values(testFeatures)
# shap.summary_plot(shap_values, testFeatures)

print("Best Model Architecture:", bestModel)

# Retrain model on full training set
training = trainingDf
trainLabels = training.iloc[:, -1]

# Isolate features once again
features = training.iloc[:, 320:341]
training.drop(training.columns[-1], axis=1, inplace=True)

# Train our new model on the whole training set
bestModel.fit(features, trainLabels)

# Remove any np.inf, -np.inf, np.nan from testing data
testingDf.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

# Isolate testing features
testFeatures = testingDf.iloc[:, 320:341]

# Run predictions on our classifier
predictions = bestModel.predict(testFeatures)

'''Voting Mechanisms - Three distinct voters that combine the 60 per minute predictions into an hourly result'''
# Convert minute predictions into hourly
n = 60
avgResult = np.average(predictions.reshape(-1, n), axis=1)
res = (avgResult > 0.1).astype(int)
utility.uniqueValueCount(res)
utility.writePredictionsToFile(res, "basic_point1_threshold.txt")

# Any instance of 1 found in numpy array
hour = predictions.reshape(-1, n)
res = []
for x in hour:
    x = x.tolist()
    if 1.0 in x:
        res.append(1.0)
    else:
        res.append(0.0)
utility.uniqueValueCount(res)
utility.writePredictionsToFile(res, "any_occurence.txt")

# Handle missing values
# testLabelsClean = testLabels.values
# removeValCount = len(testLabelsClean) % 60
# for i in range(0, removeValCount):
#     testLabelsClean = np.delete(testLabelsClean, i)
#     bestPreds = np.delete(bestPreds, i)


avgResult = np.average(predictions.reshape(-1, n), axis=1)
res = (avgResult > 0.5).astype(int)

# Basic voting CLF
voter = LinearDiscriminantAnalysis()
voter.fit(predictions.reshape(-1, n), res)
preds = voter.predict(hour)
utility.uniqueValueCount(preds)
utility.writePredictionsToFile(preds, "LDA_voter.txt")

