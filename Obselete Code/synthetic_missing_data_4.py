import numpy as np
import pickle
import pandas as pd
from Modules import utility
import scipy
import matplotlib.pyplot as plt
import heartpy

trainingDf = pd.read_pickle("Data/SMILE_Training_DataFrame.pkl")
testingDf = pd.read_pickle("Data/SMILE_Testing_DataFrame.pkl")

# Find the rows with all values that we can use as a template
subDf = trainingDf[(trainingDf.iloc[:, 340] != 0.0) & (trainingDf.iloc[:, 341] != 0.0) & (trainingDf.iloc[:, 342] != 0.0)]
searchableValues = subDf.iloc[:, 320:340]

# Find all rows where a deep feature is missing but we have the other two features to use as templates
missingDf = testingDf[(testingDf.iloc[:, 340] == 0.0) & (testingDf.iloc[:, 341] != 0.0) & (testingDf.iloc[:, 342] != 0.0)]
missingDf_1 = missingDf.iloc[:, 320:340]

for x in testingDf.index:

    # All features are present or absent (Move on)
    if testingDf.iloc[x, 340] == 1.0 and testingDf.iloc[x, 341] == 1.0 and testingDf.iloc[x, 342] == 1.0:
        pass
    elif testingDf.iloc[x, 340] == 0.0 and testingDf.iloc[x, 341] == 0.0 and testingDf.iloc[x, 342] == 0.0:
        pass
    # Otherwise synthesize values based on the remaining features
    elif testingDf.iloc[x, 340] == 1.0:
        # Start building the DataFrame we will compare for similarity against
        searchableValues = pd.DataFrame()
        tempDf = pd.DataFrame()

        # Replace feature flags
        deepMissing = False
        ecgMissing = False
        gsrMissing = False


        # print("Missing Masks:",trainingDf.iloc[x, 340], trainingDf.iloc[x, 341], trainingDf.iloc[x, 342])

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

        # Find similar values in searchable dataframe - using euclidean distance
        # print("Missing Masks:",trainingDf.iloc[x, 340], trainingDf.iloc[x, 341], trainingDf.iloc[x, 342])
        # print("index:", x)
        # print(tempDf)
        try:
            ary = scipy.spatial.distance.cdist(searchableValues, tempDf.iloc[[0]], metric='euclidean')
        # diff_df = searchableValues - tempDf.iloc[[x]].values
        # norm_df = diff_df.apply(np.linalg.norm, axis=1)

            minIndex = np.nanargmin(ary)
            testingDf.iloc[x, 343] = subDf.iloc[minIndex, 343]
        except ValueError as e:
            print(e)
        except IndexError as e:
            print(e)
    else:
        pass




    # ary = scipy.spatial.distance.cdist(searchableValues, missingDf_1.iloc[[x]], metric='euclidean')
    #
    #
    # # Find the minimum value index
    # minVal = 10
    # minIndex = 0
    # for index, y in enumerate(ary):
    #     if y < minVal:
    #         minVal = y
    #         minIndex = index
    #
    # # Replace the missing values with the values found at our similar index
    # print(missingDf.iloc[[x]])
    # missingDf.iloc[x, 0:320] = subDf.iloc[minIndex, 0:320]
    # print(missingDf.iloc[[x]])


    # Plotting code - useful for validating the values are similar
    # t = missingDf_1.iloc[[x]].values.flatten()
    # plt.plot(t)
    # plt.ylim(0, 1)
    # plt.title("Goal Values")
    # plt.show()
    # t = searchableValues.iloc[[minIndex]].values.flatten()
    # plt.plot(t)
    # plt.ylim(0, 1)
    # plt.title("Similar Values")
    # plt.show()

    # Old Pandas Based Method - provided similar values but not as close as euclidiean distance
    # diff_df = searchableValues - missingDf_1.iloc[[x]].values
    # norm_df = diff_df.apply(np.linalg.norm, axis=1)
    #
    # t = searchableValues.iloc[[339]].values.flatten()
    # plt.plot(t)
    # plt.ylim(0, 1)
    # plt.show()
    # t = missingDf_1.iloc[[x]].values.flatten()
    # plt.plot(t)
    # plt.ylim(0, 1)
    # plt.show()

name = "synthetic_euclidiean_labels_test"

trainingDf.to_pickle("Data/"+name+".pkl")
trainingDf.to_csv("Data/"+name+".csv", sep=",")


