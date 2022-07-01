import numpy as np
import pickle
import pandas as pd
from Modules import utility
import scipy
import matplotlib.pyplot as plt
import os.path

trainingFile = "Data/SMILE_Training_DataFrame.pkl"
testingFile = "Data/SMILE_Training_DataFrame.pkl"

if not os.path.exists(trainingFile) or not os.path.exists(testingFile):
    print(f"ERROR: it seems like the following files are missing:\n {trainingFile}\n {testingFile}")
    exit()


name = "synthetic_missing_data_training_euclidiean"
trainingDf = pd.read_pickle(trainingFile)

''' Uncomment to run Euclidiean feature imputation on testing data'''
# name = "synthetic_missing_data_testing_euclidiean"
# trainingDf = pd.read_pickle(testingFile)

# Find the rows with all values that we can use as a template
subDf = trainingDf[(trainingDf.iloc[:, 340] != 0.0) & (trainingDf.iloc[:, 341] != 0.0) & (trainingDf.iloc[:, 342] != 0.0)]

print("\nRunning Feature Imputation - Similarity Metric: Euclidiean Distance...")
print("This can take a little while...")
for x in trainingDf.index:

    # All features are present or absent (Move on)
    if trainingDf.iloc[x, 340] == 1.0 and trainingDf.iloc[x, 341] == 1.0 and trainingDf.iloc[x, 342] == 1.0:
        pass
    elif trainingDf.iloc[x, 340] == 0.0 and trainingDf.iloc[x, 341] == 0.0 and trainingDf.iloc[x, 342] == 0.0:
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
        if trainingDf.iloc[x, 340] == 0.0:
            # set flag to replace deep features
            deepMissing = True
        else:
            searchableValues = pd.concat([searchableValues, subDf.iloc[:, 0:321]], ignore_index=True, axis=1)
            tempDf = pd.concat([tempDf, trainingDf.iloc[[x], 0:321]], ignore_index=True, axis=1)

        # Are we missing ecg features (ecg_masking == 0.0)
        if trainingDf.iloc[x, 341] == 0.0:
            # set flag to replace ecg features
            ecgMissing = True
        else:
            searchableValues = pd.concat([searchableValues, subDf.iloc[:, 321:329]], ignore_index=True, axis=1)
            tempDf = pd.concat([tempDf, trainingDf.iloc[[x], 321:329]], ignore_index=True, axis=1)

        # Are we missing gsr features (gsr_masking == 0.0)
        if trainingDf.iloc[x, 342] == 0.0:
            # set flag to replace gsr features
            gsrMissing = True
        else:
            searchableValues = pd.concat([searchableValues, subDf.iloc[:, 329:341]], ignore_index=True, axis=1)
            tempDf = pd.concat([tempDf, trainingDf.iloc[[x], 329:341]], ignore_index=True, axis=1)

        ary = scipy.spatial.distance.cdist(searchableValues, tempDf.iloc[[0]], metric='euclidean')

        try:
            minIndex = np.nanargmin(ary)

            if deepMissing:
                # Pull deep values from similar row and replace in location
                trainingDf.iloc[x, 0:321] = subDf.iloc[minIndex, 0:321]
            if ecgMissing:
                # Pull ecg values from similar row and replace in location
                trainingDf.iloc[x, 321:329] = subDf.iloc[minIndex, 321:329]
            if gsrMissing:
                # Pull gsr values from similar row and replace in location
                trainingDf.iloc[x, 329:341] = subDf.iloc[minIndex, 329:341]

        except ValueError as e:
            print(e)
        except IndexError as e:
            print(e)


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

trainingDf.to_pickle("Data/"+name+".pkl")
trainingDf.to_csv("Data/"+name+".csv", sep=",")


