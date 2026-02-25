# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# Load datasets
logs_df = spark.read.csv("listening_logs.csv", header=True, inferSchema=True)
songs_df = spark.read.csv("songs_metadata.csv", header=True, inferSchema=True)

# Task 1: User Favorite Genres
# Join logs and metadata, group by user and genre, find the top genre for each user
user_genre_counts = logs_df.join(songs_df, "song_id") \
    .groupBy("user_id", "genre") \
    .count()

window_spec_fav = Window.partitionBy("user_id").orderBy(desc("count"))
user_fav_genre = user_genre_counts.withColumn("rank", row_number().over(window_spec_fav)) \
    .filter(col("rank") == 1) \
    .select("user_id", "genre", col("count").alias("genre_listen_count"))

print("Task 1: User Favorite Genres (Top 10)")
user_fav_genre.show(10)
user_fav_genre.write.csv("outputs/user_fav_genre", header=True, mode="overwrite")


# Task 2: Average Listen Time per User
avg_listen_time = logs_df.groupBy("user_id") \
    .agg(avg("duration_sec").alias("avg_duration"))

print("Task 2: Average Listen Time per User (Top 10)")
avg_listen_time.show(10)
avg_listen_time.write.csv("outputs/avg_listen_time", header=True, mode="overwrite")



# Task 3: Genre Loyalty Scores
# Loyalty = (Total listen time in genre / Total listen time by user) * 100
user_total_time = logs_df.groupBy("user_id").agg(sum("duration_sec").alias("total_user_time"))
genre_total_time = logs_df.join(songs_df, "song_id") \
    .groupBy("user_id", "genre") \
    .agg(sum("duration_sec").alias("genre_time"))

genre_loyalty = genre_total_time.join(user_total_time, "user_id") \
    .withColumn("loyalty_score", (col("genre_time") / col("total_user_time")) * 100)

window_spec_loyalty = Window.partitionBy("genre").orderBy(desc("loyalty_score"))
top_loyal_users = genre_loyalty.withColumn("rank", row_number().over(window_spec_loyalty)) \
    .filter(col("rank") <= 10)

print("Task 3: Top 10 Loyal Users per Genre")
top_loyal_users.select("genre", "user_id", "loyalty_score").show(20)
top_loyal_users.write.csv("outputs/genre_loyalty", header=True, mode="overwrite")


# Task 4: Identify users who listen between 12 AM and 5 AM
night_listeners = logs_df.withColumn("hour", hour(col("timestamp"))) \
    .filter((col("hour") >= 0) & (col("hour") < 5)) \
    .select("user_id").distinct()

print("Task 4: Users who listen between 12 AM and 5 AM (Count: {})".format(night_listeners.count()))
night_listeners.show(10)
night_listeners.write.csv("outputs/midnight_listeners", header=True, mode="overwrite")
