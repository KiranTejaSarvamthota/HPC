# NetflixMovieRecommendation-CF-ALS-Spark

I’ve chosen Movielens data set having user ratings for different movies. The idea is to create a recommendation system for each user based on his previous ratings. A collaborative filtering model is built to predict the ratings for the user for the movie he didn’t watch. I’ve used Normalized the ratings data and also built a scalable model.

DEMO Link:
https://drive.google.com/open?id=1XXbMKfZ1Bqdcq1eZV9e1gXSHLbQNEHzQ

create a folder named "movie" in the hadoop file system. place the input files into them. command to place the files in the HDFS

hadoop fs - put filename hdfs-file-path

Then run the ALS.py file

command

spark-submit ALS.py

after finishing executing the recommendationoutput.txt is generated where the recommendations are written.
