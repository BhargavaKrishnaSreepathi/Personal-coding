# Use this code to predict the percentage tip expected after a trip in NYC green taxi
# The code is a predictive model that was built and trained on top of the Gradient Boosting Classifer and
# the Random Forest Gradient both provided in scikit-learn

# The input:
#    spark.dataframe with columns: This should be in the same format as downloaded from the website

# The data frame go through the following pipeline:
# 1. Cleaning
# 2. Creation of derived variables
# 3. Making predictions

# The output:
#    pandas.Series, three files are saved on disk,  prediction.csv, tip_amount.csv and cleaned_data.csv respectively.

from pyspark import SparkFiles, SparkContext, SQLContext
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import pandas as pd
import pickle
from sklearn import metrics  # model optimization and valuation tools
import warnings
from pyspark.sql.types import *
from pyspark.sql.functions import *
warnings.filterwarnings('ignore')

__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"

# define a function to clean a loaded dataset
def __clean_data__(data, folder_to_be_saved_to):
    """
    This function cleans the input dataframe adata:
    . drop Ehail_fee [99% transactions are NaNs]
    . impute missing values in Trip_type
    . replace invalid data by most frequent value for RateCodeID and Extra
    . encode categorical to numeric
    . rename pickup and dropff time variables (for later use)

    input:
        data: dataframe
    output:
        dataframe
    """
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


# Function to run the feature engineering
def __engineer_features__(data, folder_to_be_saved_to):
    """
    This function create new variables based on present variables in the dataset adata. It creates:
    . Speed_mph: float, speed of the trip
    . Tip_percentage: float, target variable
    . With_tip: int {0,1}, 1 = transaction with tip, 0 transction without tip

    input:
        adata: dataframe
    output:
        dataframe
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


def __predict_tip__(transaction, folder_to_be_saved_to):
    """
    This function predicts the percentage tip expected on 1 transaction
    transaction: pandas.dataframe
    instead of calling this function immediately, consider calling it from "make_predictions"
    """
    # load models
    with open('my_classifier.pkl', 'rb') as fid:
        classifier = pickle.load(fid)
        fid.close()
    with open('my_regressor.pkl', 'rb') as fid:
        regressor = pickle.load(fid)
        fid.close()

    cls_predictors = ['payment_type', 'total_amount', 'trip_duration', 'speed_mph', 'mta_tax',
                      'extra']
    reg_predictors = ['total_amount', 'trip_duration', 'speed_mph']

    # classify transactions
    clas = classifier.predict(transaction[cls_predictors])

    # estimate and return tip percentage
    return clas * regressor.predict(transaction[reg_predictors])


def evaluate_predictions(folder_to_be_saved_to):
    """
    This looks for cleaned and predicted data set on disk and compare them
    """
    cleaned = pd.read_csv(folder_to_be_saved_to + 'cleaned_data.csv')
    predictions = pd.read_csv(folder_to_be_saved_to + 'submission.csv')
    print ("mean squared error:", metrics.mean_squared_error(cleaned.tip_percentage, predictions.predictions))
    print ("r2 score:", metrics.r2_score(cleaned.tip_percentage, predictions.predictions))


def make_predictions(data, folder_to_be_saved_to):
    """
    This makes sure that data has the right format and then send it to the prediction model to be predicted
    data: pandas.dataframe, raw data from the website
    the outputs are saved on disk: submissions and cleaned data saved as submission.csv and cleaned_data.csv respectively
    """

    print ("predicting ...")
    preds = pd.DataFrame(__predict_tip__(data), columns=['predictions'])


    preds.index = data.index
    pd.DataFrame(data.tip_percentage * data.total_amount, columns=['tip_amount']).to_csv(folder_to_be_saved_to + 'cleaned_data.csv', index=True)
    preds.to_csv(folder_to_be_saved_to + 'predictions.csv', index=True)
    tips = preds['predictions'] * data.total_amount / 100.0
    tip_amount_actual = data.tip_amount
    tips_prediction = pd.DataFrame({'tips': tips, 'tip_amount_actual': tip_amount_actual})
    tips_prediction.index = data.index

    tips_prediction.to_csv(folder_to_be_saved_to + 'tip_amount.csv', index=True)
    print ("submissions and cleaned data savdataed as submission.csv and cleaned_data.csv respectively")
    print ("run evaluate_predictions() to compare them")

if __name__ == '__main__':

    path = r"C:\Users\krish\Downloads/2017_Green_Taxi_Trip_Data.csv" # the path for the data
    folder_to_be_saved_to = r"D:\OneDrive\Career Development\Job\NTT_Data/" # the output folder

    sc = SparkContext()
    sc.addFile(path)
    sqlContext = SQLContext(sc)
    data = sqlContext.read.csv(SparkFiles.get("2017_Green_Taxi_Trip_Data.csv"), header=True, inferSchema=True)
    print("cleaning ...")
    data = __clean_data__(data, folder_to_be_saved_to)
    print ("creating features ...")
    data = __engineer_features__(data, folder_to_be_saved_to)

    data_feb = data.where(data.month == 2)
    test = data_feb.toPandas()
    make_predictions(test, folder_to_be_saved_to)
    evaluate_predictions(folder_to_be_saved_to)
