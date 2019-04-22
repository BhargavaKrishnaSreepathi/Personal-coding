
from pyspark import SparkFiles, SparkContext, SQLContext
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics  # model optimization and valuation tools
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score  # Perforing grid search
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings
from pyspark.sql.types import *
from pyspark.sql.functions import *
warnings.filterwarnings('ignore')

__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"

# define a function to clean a loaded dataset
def clean_data(data):
    """
    This function cleans the input dataframe adata:
    . drop Ehail_fee [99% transactions are NaNs]
    . replace invalid data by most frequent value for RateCodeID and Extra

    input:
        adata: Dataframe
    output:
        Dataframe

    """
    # derive time variables
    print ("deriving time variables...")

    # we will take a snippet of the data for exploration.
    data = data.withColumnRenamed('lpep_dropoff_datetime', 'Dropoff_dt')
    data = data.withColumnRenamed('lpep_pickup_datetime', 'Pickup_dt')

    data = data.withColumn('Pickup_dt', unix_timestamp('Pickup_dt', "dd/MM/yyyy hh:mm:ss a").cast(TimestampType()))
    data = data.withColumn('Dropoff_dt', unix_timestamp('Dropoff_dt', "dd/MM/yyyy hh:mm:ss a").cast(TimestampType()))

    data = data.withColumn("month", month('Pickup_dt').cast(IntegerType()))
    data = data.withColumn("hour", hour('Pickup_dt').cast(IntegerType()))
    data = data.withColumn("week_day", dayofweek('Pickup_dt').cast(IntegerType()))
    data = data.withColumn("month_day", dayofmonth('Pickup_dt').cast(IntegerType()))

    ## drop Ehail_fee: 99% of its values are NaNs
    if 'ehail_fee' in data.columns:
        data = data.drop('ehail_fee')

    ## replace all values that are not allowed as per the variable dictionary with the most frequent allowable value
    # remove negative values from Total amound and Fare_amount

    data = data.withColumn('total_amount', abs(data.total_amount).cast(FloatType()))
    data = data.withColumn('fare_amount', abs(data.fare_amount).cast(FloatType()))
    data = data.withColumn('improvement_surcharge', abs(data.improvement_surcharge).cast(FloatType()))
    data = data.withColumn('tip_amount', abs(data.tip_amount).cast(FloatType()))
    data = data.withColumn('tolls_amount', abs(data.tolls_amount).cast(FloatType()))
    data = data.withColumn('mta_tax', abs(data.mta_tax).cast(FloatType()))

    # RateCodeID
    data = data.withColumn('RatecodeID', when(col('RatecodeID') > '6', 1).otherwise(col('RatecodeID')).cast(FloatType()))

    # Extra   # 0 was identified as the most frequent value
    data = data.withColumn('extra', when(col('extra') < '0', 0).otherwise(col('extra')).cast(FloatType()))
    data = data.withColumn('extra', when(col('extra') > '1', 0).otherwise(col('extra')).cast(FloatType()))

    # Total_amount: the minimum charge is 2.5, so I will replace every thing less than 2.5 by the median 11.15 (pre-obtained in analysis)
    data = data.withColumn('total_amount', when(col('total_amount') < '2.5', 11.15).otherwise(col('total_amount')).cast(FloatType()))

    print ("Done cleaning")
    return data

def engineer_features(data):
    """
    This function create new variables based on present variables in the dataset adata. It creates:
    . Speed_mph: float, speed of the trip
    . Tip_percentage: float, target variable
    . With_tip: int {0,1}, 1 = transaction with tip, 0 transction without tip

    input:
        adata: pandas.dataframe
    output:
        pandas.dataframe
    """
    # Trip duration
    data = data.withColumn('trip_duration', ((unix_timestamp('Dropoff_dt', format="dd/MM/yyyy hh:mm:ss a") - unix_timestamp('Pickup_dt', format="dd/MM/yyyy hh:mm:ss a"))/60.0).cast(FloatType()))
    data = data.where(col('trip_duration') > 0.0)
    data = data.where(col('trip_distance') > 0.1) # making sure the distance is atleast 0.1 miles

    # create variable for Speed
    data = data.withColumn('speed_mph', (col('trip_distance') / (col('trip_duration') / 60.0)).cast(FloatType()))
    data = data.withColumn('speed_mph', when(col('speed_mph') > '200', 12.9).otherwise(col('speed_mph')).cast(FloatType()))
    data = data.withColumn('speed_mph', when(col('speed_mph') < '0', 12.9).otherwise(col('speed_mph')).cast(FloatType()))

    # create tip percentage variable
    data = data.withColumn('tip_percentage', 100 * (col('tip_amount')/col('total_amount')).cast(FloatType()))

    # create with_tip variable
    data = data.withColumn('with_tip', when(col('tip_percentage') > 0.0, 1).otherwise(0).cast(IntegerType()))

    return data


# define a function that help to train models and perform cv
def modelfit(alg, dtrain, predictors, target, scoring_method, performCV=True, printFeatureImportance=False, cv_folds=5):
    """
    This functions train the model given as 'alg' by performing cross-validation. It works on both regression and classification
    alg: sklearn model
    dtrain: pandas.DataFrame, training set
    predictors: list, labels to be used in the model training process. They should be in the column names of dtrain
    target: str, target variable
    scoring_method: str, method to be used by the cross-validation to valuate the model
    performCV: bool, perform Cv or not
    printFeatureImportance: bool, plot histogram of features importance or not
    cv_folds: int, degree of cross-validation
    """
    # train the algorithm on data
    alg.fit(dtrain[predictors], dtrain[target])
    # predict on train set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    if scoring_method == 'roc_auc':
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # perform cross-validation
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=cv_folds,
                                                    scoring=scoring_method)
        # print model report
        print ("\nModel report:")
        if scoring_method == 'roc_auc':
            print ("Accuracy:", metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
            print ("AUC Score (Train):", metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        if (scoring_method == 'mean_squared_error'):
            print ("Accuracy:", metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    if performCV:
        print ("CV Score - Mean : %.7g | Std : %.7g | Min : %.7g | Max : %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    # print feature importance
    if printFeatureImportance:
        if dir(alg)[0] == '_Booster':  # runs only if alg is xgboost
            feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        else:
            feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importe Score')
        plt.show()


# optimize n_estimator through grid search
def optimize_num_trees(alg, param_test, scoring_method, train, predictors, target):
    """
    This functions is used to tune paremeters of a predictive algorithm
    alg: sklearn model,
    param_test: dict, parameters to be tuned
    scoring_method: str, method to be used by the cross-validation to valuate the model
    train: pandas.DataFrame, training data
    predictors: list, labels to be used in the model training process. They should be in the column names of dtrain
    target: str, target variable
    """
    gsearch = GridSearchCV(estimator=alg, param_grid=param_test, scoring=scoring_method, n_jobs=2, iid=False, cv=5)
    gsearch.fit(train[predictors], train[target])
    return gsearch


def predict_tip(transaction):
    """
    This function predicts the percentage tip expected on 1 transaction
    transaction: pandas.dataframe, this should have been cleaned first and feature engineered
    """
    # define predictors labels as per optimization results
    cls_predictors = ['payment_type', 'total_amount', 'trip_duration', 'speed_mph', 'mta_tax',
                      'extra']
    reg_predictors = ['total_amount', 'trip_duration', 'speed_mph']

    # classify transactions
    clas = gs_cls.best_estimator_.predict(transaction[cls_predictors])

    # predict tips for those transactions classified as 1
    return clas * gs_rfr.best_estimator_.predict(transaction[reg_predictors])

def machine_learning_using_scikit(data, folder_to_be_saved_to):

    train = data
    # since the dataset is too big for my system, select a small sample size to carry on training and 5 folds cross validation
    train = train.loc[np.random.choice(train.index, size=100000, replace=False)]
    target = 'with_tip'  # set target variable - it will be used later in optimization

    tic = datetime.datetime.now()  # initiate the timing
    # for predictors start with candidates identified during the EDA
    predictors = ['payment_type', 'total_amount', 'trip_duration', 'speed_mph', 'mta_tax',
                  'extra']

    # optimize n_estimator through grid search
    param_test = {'n_estimators': range(30, 151, 20)}  # define range over which number of trees is to be optimized

    # initiate classification model
    model_cls = GradientBoostingClassifier(
        learning_rate=0.1,  # use default
        min_samples_split=2,  # use default
        max_depth=5,
        max_features='auto',
        subsample=0.8,  # try <1 to decrease variance and increase bias
        random_state=10)

    # get results of the search grid
    gs_cls = optimize_num_trees(model_cls, param_test, 'roc_auc', train, predictors, target)
    print(gs_cls.best_params_, gs_cls.best_score_)

    # cross validate the best model with optimized number of estimators
    modelfit(gs_cls.best_estimator_, train, predictors, target, 'roc_auc')

    # save the best estimator on disk as pickle for a later use
    with open(folder_to_be_saved_to + 'my_classifier.pkl', 'wb') as fid:
        pickle.dump(gs_cls.best_estimator_, fid)
        fid.close()

    print("Processing time:", datetime.datetime.now() - tic)

    train = train.loc[np.random.choice(train.index, size=100000, replace=False)]

    train['ID'] = train.index
    target = 'tip_percentage'
    predictors = ['total_amount', 'trip_duration', 'speed_mph']

    # Random Forest
    tic = datetime.datetime.now()
    # optimize n_estimator through grid search
    param_test = {'n_estimators': range(50, 200, 25)}  # define range over which number of trees is to be optimized

    # initiate classification model
    # rfr = RandomForestRegressor(min_samples_split=2,max_depth=5,max_features='auto',random_state = 10)
    rfr = RandomForestRegressor()  # n_estimators=100)
    # get results of the search grid
    gs_rfr = optimize_num_trees(rfr, param_test, 'neg_mean_squared_error', train, predictors, target)

    # print optimization results
    print(gs_rfr.best_params_, gs_rfr.best_score_)

    # cross validate the best model with optimized number of estimators
    modelfit(gs_rfr.best_estimator_, train, predictors, target, 'neg_mean_squared_error')

    # save the best estimator on disk as pickle for a later use
    with open(folder_to_be_saved_to + 'my_regressor.pkl', 'wb') as fid:
        pickle.dump(gs_rfr.best_estimator_, fid)
        fid.close()

    print(datetime.datetime.now() - tic)

    return gs_cls, gs_rfr

if __name__ == '__main__':
    # Download the NYC taxi dataset of 2017
    # I have stored them in the following location, change this to the location where you have it saved

    path = r"C:\Users\krish\Downloads/2017_Green_Taxi_Trip_Data.csv"
    folder_to_be_saved_to = r"D:\OneDrive\Career Development\Job\NTT_Data/" # the output folder

    sc = SparkContext()
    sc.addFile(path)
    sqlContext = SQLContext(sc)
    data = sqlContext.read.csv(SparkFiles.get("2017_Green_Taxi_Trip_Data.csv"), header=True, inferSchema=True)

    # Print the size of the dataset
    print("Number of rows:", data.count())
    print("Number of columns: ", len(data.columns))

    data = clean_data(data)

    # run the code to create new features on the dataset
    print("size before feature engineering:", data.count(), len(data.columns))
    data = engineer_features(data)
    print("size after feature engineering:", data.count(), len(data.columns))

    data_jan = data.where(data.month == 1)
    data_feb = data.where(data.month == 2)

    data_jan.toPandas().to_csv(folder_to_be_saved_to + "jan_2017.csv")
    data_feb.toPandas().to_csv(folder_to_be_saved_to + "feb_2017.csv")

    train = data_jan.toPandas()
    test = data_feb.toPandas()

    print("Optimizing the classifier...")

    gs_cls, gs_rfr = machine_learning_using_scikit(train, folder_to_be_saved_to)

    ypred = predict_tip(test)
    print("final mean_squared_error:", metrics.mean_squared_error(ypred, test.tip_percentage))
    print("final r2_score:", metrics.r2_score(ypred, test.tip_percentage))


