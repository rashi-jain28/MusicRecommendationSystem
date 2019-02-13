# MusicRecommendationSystem

#### Files used
```
For data cleaning- Recommendation System.ipynb
For ALS implementation- als.py
For KNN(cosine similarity) - knn.py
```	
	
	

#### Assumption


    The system already contains Spark, NumPy, PySpark, Python.
    All the files used are already in HDFS of Cluster and Cloudera.
	
	


#### Execution Steps

### ALS Implementation


Step 1: Run the below Recommendation_System.ipynb on jupyter notebook or iPython notebook for data data-Preparation.
		It will create PreProcessedFile.csv, which will be used for ALS algorithm implementation.
		
Step 2: Executing ALS

On DSBA Cluster:

1.hdfs dfs -mkdir /user/rjain12/cproject
2.Putting the files on dsba cluster from local. Run below commands in Cloudera local:
	scp /home/cloudera/cproject/als.py rjain12@dsba-hadoop.uncc.edu:/users/rjain12/cproject	
	scp /home/cloudera/cproject/PresprocessedFile.csv rjain12@dsba-hadoop.uncc.edu:/users/rjain12/cproject
			
3.cd cproject
4.Put the files on dsba hdfs :
	hdfs dfs -put PreProcessedFile.csv /user/rjain12/cproject
	
5. Execute the code for both the files:

Arguments:
```
		arg 1 ---->  als.py   (python code for als)
		arg 2 ----> /user/rjain12/cproject/PreProcessedFile.csv  (point to the dataset)
		arg 3 ----> 5 (user id for whom we want to give recommendation)
```
```
$ spark-submit als.py /user/rjain12/cproject/PreProcessedFile.csv 5
```


### KNN with Cosine Similarity


Step 1: Run the below Recommendation_System.ipynb on jupyter notebook or iPython notebook for data data-Preparation.
		It will create PreProcessedFile.csv, which will be used for KNN algorithm implementation.
		
Step 2: Executing KNN

On DSBA Cluster:

1.hdfs dfs -mkdir /user/rjain12/cproject
2.Putting the files on dsba cluster from local. Run below commands in Cloudera local:
	scp /home/cloudera/cproject/knn.py rjain12@dsba-hadoop.uncc.edu:/users/rjain12/cproject	
	scp /home/cloudera/cproject/PresprocessedFile.csv rjain12@dsba-hadoop.uncc.edu:/users/rjain12/cproject
			
3.cd cproject
4.Put the files on dsba hdfs :
	hdfs dfs -put PreProcessedFile.csv /user/rjain12/cproject
	
5. Execute the code for both the files:
Arguments: 
```
		arg 1 ---->  knn.py   (python code for als)
		arg 2 ----> /user/rjain12/cproject/PreProcessedFile.csv  (point to the dataset)
		arg 3 ----> 10 (user id for whom we want to give recommendation)
```
```
$spark-submit knn.py /user/rjain12/cproject/PreProcessedFile.csv 10
```
