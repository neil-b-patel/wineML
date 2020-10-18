#
# PART 1: IMPORT LIBRARIES AND MODULES
#

# IMPORT DATAFRAME SUPPORT LIBRARY
import pandas as pd

# TWO CHOICES FOR PERSISTENCE MODEL (WAYS OF SAVING MODELS)
import pickle  # PYTHON'S BUILT-IN
from joblib import dump, load  # JOBLIB'S REPLACEMENT

# IMPORT DATASETS, MODEL CHOICE UTILITIES, PREPROCESSING UTILITIES
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# IMPORT MODEL FAMILIES (i.e. SVM, LINEAR REGRESSION MODEL, etc.)
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

# IMPORT CROSS-VALIDATION UTILITIES
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# IMPORT EVALUATION METHODS FOR METRICS
from sklearn.metrics import mean_squared_error, r2_score


#
# PART 2: LOAD DATASET
#

# USE PANDA TO READ IN OUR DATASET
data = pd.read_csv('winequality-red.csv')

# THIS LOOKS MESSY! WHY DO YOU THINK?
# print(data.head)

# THAT'S RIGHT ONLY 1 COLUMN!
# COLUMNS ARE SEPARATED BY SEMICOLONS IN THE CASE OF THIS CSV
data = pd.read_csv('winequality-red.csv', sep=";")

# AH MUCH BETTER. AND LOOK, 12 COLUMNS!
# print(data.head)

# IF WE WANT TO SEE THE DIMENSIONS OR "SHAPE" OF THE DATA
# SHAPE = (ROWS, COLUMNS)
# print(data.shape)

# SUMMARIZE STATS
# print(data.describe())

# NOTE: ALTHOUGH ALL FEATURES (COLUMNS) HAVE THE SAME TYPE OF DATA, THEIR SCALES STILL DIFFER
# THUS, WE NEED TO STANDARDIZE OUR DATA


#
# PART 3: SPLIT DATA INTO TRAINING AND TEST SETS
#

y = data.quality
X = data.drop('quality', axis=1)

# SET ASIDE 20% OF DATA FOR TEST SET
# SET ARBITRARY 'RANDOM STATE' (SEED)
# STRATIFY SAMPLE BY TARGET VARIABLE TO MAKE SURE TRAINING SET LOOKS LIKE TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)


#
# PART 4: DECLARE DATA PREPROCESSING STEPS
#

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))


#
# PART 5: DECLARE HYPER PARAMETERS
#

#           MODEL PARAMETERS                      VS                        HYPER PARAMETERS
#    CAN BE LEARNED DIRECTLY FROM DATA                   STRUCTURAL INFO ABOUT MODEL, SET BEFORE TRAINING MODEL
#      I.E. REGRESSION COEFFICIENTS                             I.E. WHICH CRITERIA TO USE, MSE OR MAE


# LIST OF PARAMETERS
# print(pipeline.get_params())

hyper_parameters = {
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestregressor__max_depth': [None, 5, 3, 1]
}


#
# PART 6: TUNE MODEL WITH CROSS-VALIDATION PIPELINE
#

# WHY DO WE CROSS-VALIDATE?
    # HELPS MAXIMIZE MODEL PERFORMANCE
    # REDUCES THE CHANCE OF OVERFITTING

# WHAT IS CROSS-VALIDATION?
# USING OUR TRAINING SET, CROSS-VALIDATION JUDGES THE EFFECTIVENESS OF DIFFERENT HYPER PARAMETERS

# USE GRID SEARCH METHOD, WHICH TRIES ALL PERMUTATIONS OF HYPER PARAMETERS
# GIVE IT OUR PIPELINE, OUR HYPER PARAMETERS, AND THE NUMBER OF 'FOLDS' (THE NUMBER OF ITERATIONS TO RUN CV)
clf = GridSearchCV(pipeline, hyper_parameters, cv=10)

# FIT AND TUNE MODEL
clf.fit(X_train, y_train)

# OUR BEST HYPER PARAMETER CHOICES TURN OUT TO BE NONE AND SQRT
print(clf.best_params_)


#
# PART 7: REFIT ON TRAINING SET, IF NOT DONE BY DEFAULT
#

# FIT MODEL ON OUR TRAINING SET
# GRID SEARCH REFITS AUTOMATICALLY
# print(clf.refit)


#
# PART 8: EVALUATE MODEL ON UNSEEN DATA: THE TEST SET
#

y_pred = clf.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# FINALLY, SAVE OUR HARD-EARNED MODEL
dump(clf, 'rf_regressor.pkl')