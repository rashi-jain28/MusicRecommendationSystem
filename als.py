# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import sys
from numpy.random import rand
from numpy import matrix
import numpy as np
import csv 


# <codecell>

from pyspark import SparkContext

# <codecell>

#Parsing the Rating File
def PlaysPerTrack(line):
    list = line.replace("\n", "").split(",") #Combine all the rows of the excel files seperate them by a ,
    #print('**********************************')
    #print(list)
    if(len(list) == 6):		
    	try:
        	return int(list[3]), int(list[4]), int(list[5]),list[1] #Split on the basis of the , and get individual lists of # userIDs, trackID and plays
    	except ValueError:    #Exception Handling for if there is any problem while assigning value
        	pass

# <codecell>

#Calculating RMSE for performance evaluation   
def get_rms_error(rating_mat, track_mat, user_mat):
    val_differ = rating_mat - (track_mat * user_mat.T)
    val_differ_sq = (np.power(val_differ, 2)) / (track_mat_row * user_mat_row)
    return np.sqrt(np.sum(val_differ_sq))

# <codecell>

#For Matrix product and solving the equation to get the values
def extrapolate(x,mat,ratingMat):
    XXt=mat.T * mat
    r,c=ratingMat.shape
    for i in range(prop):
        XXt[i, i] = XXt[i,i] + lamda_val * r
    trans=mat.T
    rateTrans=ratingMat[x,:].T
    YXt=trans * rateTrans
    return np.linalg.solve(XXt,YXt)
                       

# <codecell>

if __name__ == "__main__":
    
    if len(sys.argv) !=3:
    	print >> sys.stderr, "Usage: ALS Recommendation <datafile>"
    	exit(-1)

    sc = SparkContext(appName="Recommendation_ALS")

    num_itr =  10   
    rms_val = np.zeros(num_itr)
    lamda_val = 0.001
    prop = 15
      
    #Read the csv
    #df=sc.textFile('/user/ptaneja/cproject/small.csv')
    df=sc.textFile(sys.argv[1])
    label = df.first() 
    rawdata = df.filter(lambda x: x != label)
    dataRDD1 = rawdata.map(PlaysPerTrack).cache()
        
    dataRDD = dataRDD1.filter(lambda x: x is not None)
    #collect the import rows for computation
    print('---------------------------------------------length of RDD is: %d' %dataRDD.count())
    #print(dataRDD.count())
    userIDs=dataRDD.map(lambda v:v[0] if v else None).collect()
    trackIDS=dataRDD.map(lambda v:v[1] if v else None).collect()
    plays=dataRDD.map(lambda v:v[2] if v else None).collect()
    trackNames=dataRDD.map(lambda v:v[3] if v else None).collect()
	
        
    #Create arrays for unique users, their corresponding tracks and track Names
    users=[]
    music=[]
    play_by_user={}
    trackBytrackID={}
        
    for i in range(len(userIDs)):
	print('------------------------------------------------- %d' %i)
        if not int(trackIDS[i]) in music:
            music.append(trackIDS[i])
        if not int(trackIDS[i]) in trackBytrackID:
            trackBytrackID[int(trackIDS[i])]=trackNames[i]
        if not int(userIDs[i]) in play_by_user:
            users.append(int(userIDs[i]))
            play_by_user[int(userIDs[i])]={}
            play_by_user[int(userIDs[i])][int(trackIDS[i])]=float(plays[i])
        else:
            play_by_user[int(userIDs[i])][int(trackIDS[i])]=float(plays[i])
                    
    print ('Number of Users: %d' %len(users))
    print ('Number of Tracks : %d' %len(music))
    
        
    #Creating the initial Matrix
    i=0;
    lis=[]
    for user_id in users:
	#print('------------------------------------------------- %d' %i)
        users= [0.0]*len(music)
        for product_id in range(0, len(music)):
            if user_id in play_by_user:
                if music[product_id] in play_by_user[user_id]:
                    try:
                        exist_productid=music[product_id]
                        users[product_id]=float(play_by_user[user_id][exist_productid])
                    except: 
                        continue;
        #print(users)
        lis.insert(i,[users])
        i+=1    
    
        
        
    lisarray=np.asarray(lis)
    rating_mat= np.matrix(lisarray)
    rating_mat.shape
    num_row,num_col = rating_mat.shape  
    broad_rating_mat = sc.broadcast(rating_mat)
    
    
    
    '''
    ALS Implementation
    '''
    #Broadcast the rating, user matrix and track matrix to all the nodes
    track_mat = matrix(rand(num_row, prop))
    broad_track = sc.broadcast(track_mat)
            
    user_mat = matrix(rand(num_col, prop))
    broad_user = sc.broadcast(user_mat)
           
    track_mat_row,track_mat_col = track_mat.shape
    user_mat_row,user_mat_col = user_mat.shape
        
    # iterating until track matrix and user matrix converges
    for i in range(0,num_itr):
	print('------------------------------------------------- %d' %i)
        track_mat = sc.parallelize(range(track_mat_row)).map(lambda x:extrapolate(x,broad_user.value,broad_rating_mat.value)).collect()
        broad_track = sc.broadcast(matrix(np.array(track_mat)[:, :]))
        
        user_mat = sc.parallelize(range(user_mat_row)).map(lambda x:extrapolate(x,broad_track.value,broad_rating_mat.value.T)).collect()
        broad_user = sc.broadcast(matrix(np.array(user_mat)[:, :]))
        
        rms_error_val = get_rms_error(rating_mat, matrix(np.array(track_mat)), matrix(np.array(user_mat)))
        rms_val[i] = rms_error_val
    fin_user_mat = np.array(user_mat).squeeze()
    fin_track_mat = np.array(track_mat).squeeze()
    fin_out = np.dot(fin_track_mat,fin_user_mat.T)
        
        
    # For Initializing the weights matrix 
    weight_mat = np.zeros(shape=(num_row,num_col))
    for r in range(num_row):
        for j in range(num_col):
            if rating_mat[r,j]>= 0.5:
                weight_mat[r,j] = 1.0
            else:
                weight_mat[r,j] = 0.0
                    
    
    # subtract the rating that user has rated
    rate_max=5
    #print(fin_out )
    product_recom = np.argmax(fin_out - rate_max * weight_mat,axis =1)
        
        
    # To Predict the best track for each user
    u=int(sys.argv[2])
    r = product_recom.item(u)
    rp=music[r]
    trNme=trackBytrackID[rp]
    p = fin_out.item(u,r)
            
    print ('For user %s recommended track is \"%s\" with id %s' %(u,trNme,rp) )
                
        
    print ""
    print "-----------Performance Evaluation--------"
    print "All RMSE values after each iteration: ",rms_val    
    print "Avg RMSE---- ",np.mean(rms_val)
    sc.stop()

# <codecell>


# <codecell>


