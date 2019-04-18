# 1. Problem Statement
# Use GYC Green Taxi data to implement a machine learning system to predict the expected
# tip amount for a trip.
# 2. Data
# 2017 GYC Green Taxi Data:
# https://data.cityofnewyork.us/browse?q=2017%20Green%20Taxi%20Trip%20Data&sortBy=relevance
# You may use the Jan 2017 data for your model training and Feb 2017 data for model
# evaluation.

# from pyspark import SparkContext, SparkConf
# import sys
# import time
#
# sc = SparkContext()
# data = sc.textFile(r"C:\Users\krish\Downloads\2017_Green_Taxi_Trip_Data.csv")
# print (type(data))
# print (data)
#
# data.take(5)

#from pyspark.sql import SQLContext
# url = "https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/adult.csv"
from pyspark import SparkFiles, SparkContext, SQLContext

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

url = r"C:\Users\krish\Downloads/2017_Green_Taxi_Trip_Data.csv"

sc =SparkContext()
sc.addFile(url)
sqlContext = SQLContext(sc)
df = sqlContext.read.csv(SparkFiles.get("2017_Green_Taxi_Trip_Data.csv"), header=True, inferSchema= True)
# df = sqlContext.read.csv(SparkFiles.get("adult.csv"), header=True, inferSchema= True)
print (df.limit(10).toPandas())
print (df.columns)
print (df.count())
print (df.describe().show())

df_jan = df.filter(df("lpep_pickup_datetime").lt(lit("2017-01-01")))
print (df_jan.count())

# dfArray = [df.where(df.ID == x) for x in listids]