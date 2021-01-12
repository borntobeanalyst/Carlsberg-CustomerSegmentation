# Carlsberg Customer Segmentation

This project was held by London Business School London LAB 2020 programme which fostered cooperative projects between master's students and international corporations. 

Our team (where I led the international team of 6) partnered with Carlsberg, a Danish international brewer founded in 1847, which requested a data consulting with its real-time beer consumption data. Carlsberg group has launched new digital draughtmaster system to track the consumption of the beer of their own brands in different locations. Hereby, I only used the data of Italian outlets served by Carlsberg group to discover the newest customer segmentations and following customer acquisition strategy.

Since I can't expose the details of the dataset itself, I did not upload the original csv file and just shared the code to help you get the idea of the clustering methods (PAM) and the recommendation & prediction (lstm) algorithm using python with real-time data.

# Clustering Methods (PAM)

There are several ways to create clusters in machine learning area. The examples are: Hierarchical clustering, K-means clustering, K-nearest neighbors(KNN) algorithm, and PAM clustering.

Firstly, our team chose trying K-means clustering (unsupervised clustering) as we had not enough data to train the algorithm and labeled customer segmentations yet (supervised clustering). However, we soon realized that K-means clustering does not support the categorical variables and uses and changed the clustering method to PAM(Partioning Around Medoids) clustering which supports both numerical and categorical variables.
