import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data=pd.read_csv('./Mall_Customers.csv')
print(customer_data.head()) #print the first 5 rows
print(customer_data.shape) #prints the total number of rows and columns
print(customer_data.info())
print(customer_data.isnull().sum())

X=customer_data.iloc[:,[3,4]].values # taking all the rows of the third and 4th columns (to match our segmentation)
 # to choose the correct number of clusters we need for the K means , we need to use the WCSS formula
 # we will then use the Elbow method to find the corresponding number of K clusters

wcss=[]
for i in range(1,21):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=40)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plot the find the elbow point
sns.set()
plt.plot(range(1,21),wcss)
plt.title('The elbow point graph')
plt.xlabel('# of clusters')
plt.ylabel('WCSS')
plt.show()

# after plotting we found that the elbow point is at 5 clusters

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=40)

Y=kmeans.fit_predict(X) # gives for each data point in X a corresponding label to which cluster it corresponds (from 0 to 4 which are the labels of the 5 clusters)
print(Y)

#plot the graph of the clusters

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50,c='green',label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50,c='blue',label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50,c='black',label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50,c='purple',label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50,c='red',label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='white', label='centroid')
plt.title('Final result')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.show()