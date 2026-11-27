#import

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

#config et builder session spark

spark = (
    SparkSession.builder
    .master("local[*]")                             # use all cpu cores
    .config("spark.driver.memory", "8g")            # lots of memeory available
    .config("spark.sql.shuffle.partitions", "16")  
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

DATA_DIR = Path(__file__).resolve().parent / "ml-32m" / "ml-32m"
path_movies_csv = str(DATA_DIR / "movies.csv")
path_ratings_csv = str(DATA_DIR / "ratings.csv")

df_movies = spark.read.csv(path_movies_csv, header=True, inferSchema=True)
df_ratings = spark.read.csv(path_ratings_csv, header=True, schema="userId INT, movieId INT, rating FLOAT, timestamp LONG")

# drop useless column

df_ratings = df_ratings.select("userId", "movieId", "rating").cache()

#print info

"""df_movies.show()
df_ratings.show()

df_movies.printSchema()
df_ratings.printSchema()"""

#model configuration

als = ALS(maxIter=10, regParam=0.1, rank=5, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True,
          numUserBlocks=16, numItemBlocks=16)  # from 10 to 16, all threads are used

df_ratings_train, df_ratings_test = df_ratings.randomSplit([0.8,0.2], 42)

als_model = als.fit(df_ratings_train)

preds = als_model.transform(df_ratings_test).cache()  # keep in cache

preds.show(5)


#evaluation

eval = RegressionEvaluator(metricName='rmse', labelCol="rating", predictionCol="prediction")

rmse = eval.evaluate(preds)
r2 = eval.evaluate(preds, {eval.metricName: 'r2'})
mae = eval.evaluate(preds, {eval.metricName: 'mae'})
var = eval.evaluate(preds, {eval.metricName: 'var'})

print(f"The model evaluation : \nrmse = {rmse}\nr2 = {r2}\nmae = {mae}\nvar = {var}\n")

# Hyperparameters tuning

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

param_grid = ParamGridBuilder().addGrid(
    als.rank, [5,10,15,20]
    ).addGrid(
        als.maxIter, [10]
    ).addGrid(
        als.regParam, [0.01,0.05,0.15,0.2]
    ).build()

# split 80/20
tvs = TrainValidationSplit(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=eval,
    trainRatio=0.8,
    seed=42
)

# tuning on 1/5 of the csv to minimize time
df_tuning = df_ratings_train.sample(fraction=0.2, seed=42).cache()

als_model = tvs.fit(df_tuning)

#get rmse scores and parametors to compare

rmse_scores = als_model.validationMetrics
param_maps = als_model.getEstimatorParamMaps()

model_acc = []

#iteration to store the comparison

for param_maps, rmse in zip(param_maps, rmse_scores):
    params = {param.name: value for param, value in param_maps.items()}
    params['rmse'] = rmse
    model_acc.append(params)

#visualisationn

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

#dataframing the model_acc

df_rating_acc = pd.DataFrame(model_acc)

#griding the dataframe

heatmap_data = df_rating_acc.pivot(index='regParam', columns='rank', values='rmse')

#ploting the grid

plt.figure(figsize=(8,6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='viridis', cbar_kws={'label': 'RMSE'})
plt.title('Heatmap of RMSE by Rank and regParam')
plt.xlabel('Rank')
plt.ylabel('Regularization Parameter (regParam)')
plt.show()