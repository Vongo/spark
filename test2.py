from math import sqrt

from numpy import array

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

DATA_PATH = "/home/adelmarre/Desktop/Projects/Others/spark/data/ours/test.csv"

conf = SparkConf().setAppName("TEST")
conf = conf.setMaster("local[*]")
sc = SparkContext(conf=conf)
data = sc.textFile(DATA_PATH)
parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10,
        runs=10, initializationMode="random")

print "yolo"
