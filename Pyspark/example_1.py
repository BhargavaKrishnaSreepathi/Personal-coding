from pyspark import SparkContext
from pyspark import SparkConf


sc = SparkContext()
data = sc.read.csv('Pyspark/test.csv')