
# Implementation of Movie recommendation system on spark
from __future__ import print_function

import sys
import itertools
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkConf, SparkContext
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
import time
start = time.time()



def parseRating(line):
    """
    Parses a rating record in MovieLens format userId,movieId,rating,timestamp .
    """
    fields = line.strip().split(",")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId,movieTitle .
    """
    fields = line.strip().split(",")
    return int(fields[0]), fields[1]





def rmse(R, usermatrix, moviematrix):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    Rp = np.dot(usermatrix,moviematrix.T)
    err = R - Rp
    errsq = np.power(err, 2)
    mean = (np.sum(errsq))/(M * U)
    return np.sqrt(mean)
    
   


# function for computing the values and updating them
def update(i, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)
    
# function for normalize the user ratings 
def normalize(data):
    data = np.array(data)
    return (data-np.min(data))/(np.max(data) - np.min(data))


	
LAMBDA = 0.01   # regularization
np.random.seed(42)	
	
if __name__ == "__main__":

    print("Running Movie Recommendation system using Own ALS")

    #conf = SparkConf().setAppName("MovieLensALS").set("spark.driver.memory", "4g")
    conf = SparkConf()\
    .setAppName("MovieLensWithALS")\
    .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    
    # load ratings and movie titles

    movieLensHomeDir = "/user/movie4"

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.csv")).map(parseRating)
    print(ratings)




    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.csv")).map(parseMovie).collect())
    movie_ids = movies.keys()
    #print(movie_ids)
    movie_ids = np.sort(movie_ids)
    #print("hhhhh")
    #print(movie_ids)
    #print(movie_ids.shape)
    #print(ratings.count())
    

    #numRatings = 100
    numRatings = ratings.count()
    #print(numRatings)
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()
    #print(numUsers)
    #print(numMovies)
    print ("Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies))

    numPartitions = 5
    #print(ratings)
    
    training = ratings.values().repartition(numPartitions).cache()
    #print(training)

    
    mat=np.zeros(5931640)
    
    a_list =training.collect()

    a_array = np.array(a_list)
    #print(a_array)
    rows, row_pos = np.unique(a_array[:, 0], return_inverse=True)
    cols, col_pos = np.unique(a_array[:, 1], return_inverse=True)


    pivot_table = np.zeros((len(rows), len(cols)), dtype=a_array.dtype)
    pivot_table[row_pos, col_pos] = a_array[:, 2]

    t_matrix = np.matrix(pivot_table)

    #t_matrix_norm = np.zeros((610,9724))
    t_matrix_norm = np.zeros((numUsers,numMovies))

    for i in range(numUsers):
        t_matrix_norm[i] = normalize(t_matrix[i])

    #print(t_matrix_norm)
    t_matrix_norm = np.asmatrix(t_matrix_norm)
    #print(t_matrix_norm.shape)
    #print(type(t_matrix_norm))
		
    U =  numUsers # number of users
    M =  numMovies # number of movies
    F =  25
    ITERATIONS = 10
    partitions = 5

    R = t_matrix_norm # Rating matrix
    W = R>0.5 # Initializing the weighted Matrix
    #print(W)
    W[W == True]= 1
    W[W == False]= 0
    #print(W)

    # Initializing the Factors
    usermatrix = matrix(rand(U, F)) 
    moviematrix = matrix(rand(M, F))
    # Broadcasting the Matrices
    Rb = sc.broadcast(R)
    userb = sc.broadcast(usermatrix)
    movieb = sc.broadcast(moviematrix)

    for i in range(ITERATIONS):
        # parallelizing the computation
        usermatrix = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, movieb.value, Rb.value)) \
               .collect()

        
        # arranging it into a matrix 
        usermatrix = matrix(np.array(usermatrix)[:, :, 0])
        #print("ms after")
        #print(usermatrix.shape)
        # Broadcasting the matrix
        userb = sc.broadcast(usermatrix)
        # parallelizing the computation
        moviematrix = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, userb.value, Rb.value.T)) \
               .collect()
        # arranging into a matrix form
        moviematrix = matrix(np.array(moviematrix)[:, :, 0])

        # Broadcasting the matrix
        movieb = sc.broadcast(moviematrix)
        # getting the error rate
        error = rmse(R, usermatrix, moviematrix)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % error)
    
    
    recommendation=np.dot(usermatrix,moviematrix.T)

    output=open("recommendationsoutput.txt",'w')

    #Giving Recommendations 
    for i in range(U):
        #Returns the indices that would sort an array.
        indices = np.array(np.argsort(recommendation[i,:]))
        #Return a copy of the array collapsed into one dimension
        indices = indices.flatten()
        #print("hello")
        #print(indices)
        #print(indices.shape)
        number_of_recs = 5
        string_recommend="-------------------Recommendations for user"+"  "+str(i + 1)+"--------------------------"+"\n"
        for index in indices[::-1]:
            if ~W[i, index]:
                string_recommend=string_recommend+"Movie title"+" :  "+movies[movie_ids[index]]+","+"Movie ID"+" :  "+str(movie_ids[index])+","+"Predicted Rating: "+str(recommendation[i,index])+"\n"
                number_of_recs -= 1
                # recommended_movies.append(index)
            if number_of_recs == 0:
                break
    	final_string=string_recommend.encode('utf-8', 'ignore')

        output.write(final_string)
    end = time.time()
    print("total running time = %.2f minutes" % ((end - start)/60))
    sc.stop()
