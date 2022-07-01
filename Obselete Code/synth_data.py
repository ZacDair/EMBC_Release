import numpy as np
import pickle
import pandas as pd
from Modules import utility
import scipy
import matplotlib.pyplot as plt

trainingDf = pd.read_pickle("Data/synthetic_missing_data_training.pkl")
testingDf = pd.read_pickle("Data/SMILE_Testing_DataFrame.pkl")

# Find the rows with all values that we can use as a template
subDf = trainingDf[(trainingDf.iloc[:, 340] != 0.0) & (trainingDf.iloc[:, 341] != 0.0) & (trainingDf.iloc[:, 342] != 0.0)]

for x in testingDf.index:

    # All features are present or absent (Move on)
    if testingDf.iloc[x, 340] == 1.0 and testingDf.iloc[x, 341] == 1.0 and testingDf.iloc[x, 342] == 1.0:
        pass
    elif testingDf.iloc[x, 340] == 0.0 and testingDf.iloc[x, 341] == 0.0 and testingDf.iloc[x, 342] == 0.0:
        pass
    # Otherwise synthesize values based on the remaining features
    else:
        # Start building the DataFrame we will compare for similarity against
        searchableValues = pd.DataFrame()
        tempDf = pd.DataFrame()

        # Replace feature flags
        deepMissing = False
        ecgMissing = False
        gsrMissing = False

        # Are we missing deep features (deep_masking == 0.0)
        if testingDf.iloc[x, 340] == 0.0:
            # set flag to replace deep features
            deepMissing = True
        else:
            searchableValues = pd.concat([searchableValues, subDf.iloc[:, 0:321]], ignore_index=True, axis=1)
            tempDf = pd.concat([tempDf, testingDf.iloc[[x], 0:321]], ignore_index=True, axis=1)

        # Are we missing ecg features (ecg_masking == 0.0)
        if testingDf.iloc[x, 341] == 0.0:
            # set flag to replace ecg features
            ecgMissing = True
        else:
            searchableValues = pd.concat([searchableValues, subDf.iloc[:, 321:329]], ignore_index=True, axis=1)
            tempDf = pd.concat([tempDf, testingDf.iloc[[x], 321:329]], ignore_index=True, axis=1)

        # Are we missing gsr features (gsr_masking == 0.0)
        if testingDf.iloc[x, 342] == 0.0:
            # set flag to replace gsr features
            gsrMissing = True
        else:
            searchableValues = pd.concat([searchableValues, subDf.iloc[:, 329:341]], ignore_index=True, axis=1)
            tempDf = pd.concat([tempDf, testingDf.iloc[[x], 329:341]], ignore_index=True, axis=1)

        ary = scipy.spatial.distance.cdist(searchableValues, tempDf.iloc[[0]], metric='euclidean')

        # Find the minimum value index
        minVal = 10
        minIndex = 0
        for index, y in enumerate(ary):
            if y < minVal:
                minVal = y
                minIndex = index

        if deepMissing:
            # Pull deep values from similar row and replace in location
            trainingDf.iloc[x, 0:321] = subDf.iloc[minIndex, 0:321]
        if ecgMissing:
            # Pull ecg values from similar row and replace in location
            trainingDf.iloc[x, 321:329] = subDf.iloc[minIndex, 321:329]
        if gsrMissing:
            # Pull gsr values from similar row and replace in location
            trainingDf.iloc[x, 329:341] = subDf.iloc[minIndex, 329:341]