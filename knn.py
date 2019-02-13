# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import sys
#spark_home = os.environ.get('SPARK_HOME', None) 
#if not spark_home:
#    raise ValueError('SPARK_HOME environment variable is not set') 
#sys.path.insert(0, os.path.join(spark_home, 'python')) 
#sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.9-src.zip')) 
#execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))


import numpy as np
import math
import csv 
from pyspark import SparkContext

#Parsing the Rating File
def Rating(line):
    items = line.replace("\n", "").split(",")
    if(len(items) == 6):
    	try:
		## selecting the userID, trackId and the rating from the csv file 
    		return int(items[3]), int(items[4]), int(items[5])  
    	except ValueError:
        	pass
	
		
#Parsing the Track file
def TrackName(line):
	
    items = line.replace("\n", "").split(",") 
    if(len(items) == 6):
	## selecting the track id and the track Name from the csv file
    	try:
    		return int(items[4]), items[1]
    	except ValueError:
        	pass


def Calculate_MeanRating(userRatingGroup):
    User_ID = userRatingGroup[0]
    Rating_Sum = 0.0
    Rating_Count = len(userRatingGroup[1])
    if Rating_Count == 0:
        return (User_ID, 0.0)
    for item in userRatingGroup[1]:
        Rating_Sum += float(item[1])
    return (User_ID, 1.0 * Rating_Sum / Rating_Count)

def UserAvg_broadcast(sContext, UTrain_RDD):
    UserRatingAverage_List = UTrain_RDD.map(lambda x: Calculate_MeanRating(x)).collect()  
    UserRatingAverage_Dict = {}
    for (user, avgscore) in UserRatingAverage_List:
        UserRatingAverage_Dict[user] = avgscore       
    URatingAverage_BC = sContext.broadcast(UserRatingAverage_Dict)   
    return URatingAverage_BC

def UserTrackRatings(userRatingGroup):   
    UserID = userRatingGroup[0]   
    tracksList = [item[0] for item in userRatingGroup[1]]   
    ratingList = [item[1] for item in userRatingGroup[1]]  
    return (UserID, (tracksList, ratingList))

def UserTrackRatings_broadcast(SContext, TrainRDD):
    userTrackList = TrainRDD.map(lambda x: UserTrackRatings(x)).collect()
    userTrackDict = {}
    for (user, tupleList) in userTrackList:
        userTrackDict[user] = tupleList
    return (SContext.broadcast(userTrackDict))

def ConstructRating(tuple1, tuple2):
    ratingpair = []
    i, j = 0, 0
    user1, user2 = tuple1[0], tuple2[0]
    
    #Storing the track lists for two users
    user1TrackList = sorted(tuple1[1])   
    user2TrackList = sorted(tuple2[1])    
    
    #iterating between the two user tracks lists
    while i < len(user1TrackList) and j < len(user2TrackList): 
        if user1TrackList[i][0] < user2TrackList[j][0]:
            i += 1															
        elif user1TrackList[i][0] == user2TrackList[j][0]: #append the ratings for the common tracks.
            ratingpair.append((user1TrackList[i][1], user2TrackList[j][1]))	
	    i += 1
            j += 1
        else:
            j += 1
    return ((user1, user2), ratingpair)


# --------------------------------------Cosine_Similarity-------------------------------------------#
# This function calculates the cosine similarity for two user pairs.
# Input : Output of ConstructRating function - ((user1, user2), ratingpair)
# Output : (userIDs, (cosine similarity, count of common tracks))

def Cosine_Similarity(tup):   
    numerator = 0.0
    a, b, count = 0.0, 0.0, 0
    for our_rating_pair in tup[1]:
        numerator += our_rating_pair[0] * our_rating_pair[1] 
        a += (our_rating_pair[0]) ** 2 
        b += (our_rating_pair[1]) ** 2	
        count += 1
    denominator = math.sqrt(a) * math.sqrt(b)
    cosine = (numerator / denominator) if denominator else 0.0
    return (tup[0], (cosine, count)) 

    
# --------------------------------------User_GroupBy-------------------------------------------#
# This function groups the records by userID. 
# Input : Output of Cosine_Similarity function - (tup[0], (cosine, count))
# Output : (user1,(all the users, corresponding cos_simi, corresponding common tracks match count))

def User_GroupBy(record):
    return [(record[0][0], (record[0][1], record[1][0], record[1][1])), 
            (record[0][1], (record[0][0], record[1][0], record[1][1]))]


# --------------------------------------SimilarUser_pull-------------------------------------------#
# Input : it takes the userID, cosine cos_simi and the number of neighbors as input
# Output : returns the corresponding number of neighbors.

def SimilarUser_pull(user, records, k = 200): 
    neighborList = sorted(records, key=lambda x: x[1], reverse=True) #take in x and return the next values of neighbour
    neighborList = [x for x in neighborList if x[2] > 9]	#filter out those whose count is small 
    return (user, neighborList[:k])


# --------------------------------------UserNeighbourBroadcast-------------------------------------------#
# This function will broadcast the userNeighborRDD value

def UserNeighbourBroadcast(sContext, neighbor):
    userNeighborList = neighbor.collect()
    userNeighbor = {}
    for user, simrecords in userNeighborList:
        userNeighbor[user] = simrecords			#making a dicionary of user and corresponding neighbourlist
    neighbourBroadcast = sContext.broadcast(userNeighbor)
    return neighbourBroadcast


# --------------------------------------CalculatingError-------------------------------------------#
# Taking in actual and predicted RDDs as input and calculating RMSE and MSE.

def CalculatingError(predictedRDD, actualRDD):
    #initial transformation and joining the RDD
    predictedReformattedRDD = predictedRDD.map(lambda rec: ((rec[0], rec[1]), rec[2])) #Getting the necessary columns for error calculation
    actualReformattedRDD = actualRDD.map(lambda rec: ((rec[0], rec[1]), rec[2]))
    joinedRDD = predictedReformattedRDD.join(actualReformattedRDD) #Joining the necessary columns for both predictedRDD and actual RDD together
    #Calculating the Errors
    squaredErrorsRDD = joinedRDD.map(lambda x: (x[1][0] - x[1][1])*(x[1][0] - x[1][1]))
    totalSquareError = squaredErrorsRDD.reduce(lambda v1, v2: v1 + v2)
    numRatings = squaredErrorsRDD.count()	#ratings count
    return (math.sqrt(float(totalSquareError) / numRatings))


# --------------------------------------Prediction-------------------------------------------#
# this function predicts the rating.
# Input - the validationRDD, the neighbor dict whic has the user cosine similarity and corresponding count and Ids, average rating of each user and the number of neighbors

def Prediction(tup, neighborDict, userTrackDict, avgDict, topK):
   user, track = tup[0], tup[1] #getting the userID and trackid
   avgrate = avgDict.get(user, 0.0)
   c = 0
   simsum = 0.0 #Sum of cos_simi
   WeightedRating_Sum = 0.0
   neighbors = neighborDict.get(user, None)
   if neighbors:
       for record in neighbors:
           if c >= topK:	#if count is more than the number of neighbours
               break
           c += 1
           tracklistpair = userTrackDict.get(record[0])
           if tracklistpair is None:
               continue
           index = -1
           try:
               index = tracklistpair[0].index(track)
           except ValueError:# if error, then this neighbor hasn't rated the track yet
               continue
           if index != -1:
               neighborAvg = avgDict.get(record[0], 0.0)
               simsum += abs(record[1])
               WeightedRating_Sum += (tracklistpair[1][index] - neighborAvg) * record[1]
   predRating = (avgrate + WeightedRating_Sum / simsum) if simsum else avgrate
   return (user, track, predRating)
from collections import defaultdict


# --------------------------------------Neighborhood_size-------------------------------------------#
# this function is used to invoke the previous error calculation function and depending on the max number of neighbors and step size, 
# it iterates and finds the corresponding error for all those number of pairs.

def Neighborhood_size(predicted_RDD, validate_RDD, userNeighborDict, UserTrackDict, UserRatingAverage_Dict, K_Range):
    errors = [0] * len(K_Range)
    err= 0
    for k in K_Range:
        predictedRatingsRDD = predicted_RDD.map(
            lambda x: Prediction(x, userNeighborDict, UserTrackDict, UserRatingAverage_Dict, k)).cache()
        errors[err] = CalculatingError(predictedRatingsRDD, validate_RDD)
        err+= 1
    return errors


# --------------------------------------Final_recommend-------------------------------------------#
def Final_recommend(user, neighbors, userTrackDict, k = 200, n = 5): 
    simSumDictionary = defaultdict(float)
    weightedSumDictionary = defaultdict(float)
    for (neighbor, simScore, numCommonRating) in neighbors[:k]:
        tracklistpair = userTrackDict.get(neighbor)
        if tracklistpair:
            for index in range(0, len(tracklistpair[0])):
                trackID = tracklistpair[0][index]
                simSumDictionary[trackID] += simScore
                weightedSumDictionary[trackID] += simScore * tracklistpair[1][index]
    candidates = [(tID, 1.0 * wsum / simSumDictionary[tID]) for (tID, wsum) in weightedSumDictionary.items()]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return (user, candidates[:n])


def BroadcastTrackListDictBroadcast(sContext, movRDD):
    TrackNameList = movRDD.collect()
    TrackNamesDictionary = {}
    for (trackID, pname) in TrackNameList:
        TrackNamesDictionary[trackID] = pname
    return (sc.broadcast(TrackNamesDictionary))


def TrackNames(user, records, namedictionary):
    tracklist = []
    for record in records:
        tracklist.append(namedictionary[record[0]])
    return (user, tracklist)


if __name__ == "__main__":
    if len(sys.argv) !=3:
   	print >> sys.stderr, "Usage: linreg <datafile>"
    	exit(-1)
    sc = SparkContext(appName="KNN")
    #Reading the Data
    #input_file = sc.textFile('/user/team1/project/input/small1.csv')
    input_file = sc.textFile(sys.argv[1])
    
    #Removing the headers from the file
    file_header = input_file.first()
    input_file = input_file.filter(lambda x: x != file_header) 
    dataRDD1 = input_file.map(Rating).cache()
    dataRDD = dataRDD1.filter(lambda x: x is not None)
    trackRDD1 = input_file.map(TrackName).cache()
    trackRDD = trackRDD1.filter(lambda x: x is not None)
    #Splitting the data into 70% training and 30% testing dataset
    training_RDD, testing_RDD = dataRDD.randomSplit([7,3])
    PredictionRDD = testing_RDD.map(lambda x: (x[0], x[1])) #not including the target variable rating
    TrainUserRating_RDD = training_RDD.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().cache().mapValues(list)

    UserRatingAverage = UserAvg_broadcast(sc, TrainUserRating_RDD)
    UserTrackRatingList = UserTrackRatings_broadcast(sc, TrainUserRating_RDD)
    cartesianUser_RDD = TrainUserRating_RDD.cartesian(TrainUserRating_RDD)
    
    #taking unique user pairs from all user pairs combination
    UserPairs = cartesianUser_RDD.filter(lambda x: x[0] < x[1])
    
    #invoking the cosine function and other RDD transformation functions
    UserPairActual = UserPairs.map(lambda x: ConstructRating(x[0], x[1]))
    SimiliarUserRDD = UserPairActual.map(lambda x: Cosine_Similarity(x))
    SimiliarUserGroupRDD = SimiliarUserRDD.flatMap(lambda x: User_GroupBy(x)).groupByKey()
    UserNeighborhood_RDD = SimiliarUserGroupRDD.map(lambda x: SimilarUser_pull(x[0], x[1], 2))
    UserNeighborhood_BC = UserNeighbourBroadcast(sc, UserNeighborhood_RDD)
    
    ErrorValue = [0]
    
    #K_range is the starting number of neighbors and the ending number and the step size
    K_Range = range(10, 130, 10)
    e = 0
    ErrorValue[e] = Neighborhood_size(PredictionRDD, testing_RDD, UserNeighborhood_BC.value, UserTrackRatingList.value, UserRatingAverage.value, K_Range)
    
    print('Error values are %s' %ErrorValue)
    UserNeighborhood_RDD.map(lambda x: (x[1])).mapValues(list)
    RecommendedTracksForUser = UserNeighborhood_RDD.map(lambda x: Final_recommend(x[0], x[1], UserTrackRatingList.value))
    TrackNameDictionary = BroadcastTrackListDictBroadcast(sc, trackRDD)
    RecommendationForUser = RecommendedTracksForUser.map(lambda x: TrackNames(x[0], x[1], TrackNameDictionary.value))
    position = int(sys.argv[2])
	
   
    tracks = RecommendationForUser.filter(lambda x:x[0]==position).collect()
    print ('For user %s recommended track is \"%s\"' %(position,tracks) )	
    #print(RecommendationForUser.filter(lambda x:x[0]==position).collect())
    sc.stop()
# <codecell>

 

