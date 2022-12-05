#-------------------------------------------------------------------------
# AUTHOR: Julian Rowe
# FILENAME: clustering.py
# SPECIFICATION: Read data from training_data.csv and run k-means by plotting
# the values and calculating the Homogeneity score.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:64]

#run kmeans testing different k values from 2 until 20 clusters
k_plot = []
coefficients = []
max_coefficient = 0
k_with_max_silhouette_cofficient = KMeans

for k in range(2, 21):

     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
     k_plot.append(k)
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
     coefficients.append(silhouette_coefficient)
     if max_coefficient < coefficients[-1]:
          max_coefficient = coefficients[-1]
          k_with_max_silhouette_cofficient = k

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.scatter(k_plot, coefficients, alpha=0.5)
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header = None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
number_of_samples = len(df)
labels = np.array(df.values).reshape(1, number_of_samples)[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())