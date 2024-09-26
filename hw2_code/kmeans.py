
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np

class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        numRows = self.points.shape[0]  # Number of data points
        randomRows = np.random.choice(numRows, self.K, replace=False)  # Randomly select K indices 
        self.centers = self.points[randomRows, :]  # Fancy Indexing
        return self.centers
    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """


        randomPoint = self.points[np.randint(0, len(self.points[0]))] 

        #Maxmimize the pairwise distance:: pairwise_dist(x, y)
        selectedRows = [randomPoint]
        while (len(selectedRows) < self.K):
            distances = pairwise_dist(self.points, selectedRows)
            minDistances = np.min(distances, axis = 1)
            newRow = np.argmax(minDistances)
            selectedRows.append(newRow)

        self.centers = selectedRows

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison. 
        """        

        #self.centers is the centers
        #self.points is all of the points
        #pairwise_dist(x, y)

        distances = pairwise_dist(self.centers, self.points) #return a nx2 array of all the distances
        self.assignments = np.argmin(distances, axis = 0)
        return self.assignments

        #Did not get edge case of np.sqrt() edge case

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """

        #I assume that each cluster center is based off the mean of its points
        
        #initialize new clusters
        newCenters = np.zeros((self.K, self.centers.shape[1]), dtype=float)
        
        for clusterIndex in range(self.K):
            #compute the mean of all points in the cluster based off the assignments from before
            pointsInCluster = self.points[clusterIndex == self.assignments]
            
            
            #update cluster
            newCenter = np.mean(pointsInCluster, axis = 0)
            newCenters[clusterIndex] = newCenter

        self.centers = newCenters
        return self.centers

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and its respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        total_loss = 0
        for clusterIndex in range(self.K):
            # Get all the points belonging to this cluster based on assignments
            pointsInCluster = self.points[self.assignments == clusterIndex]

            # If there are no points in the cluster, continue
            if pointsInCluster.shape[0] == 0:
                continue

            # Calculate squared distances: ||x - mu||^2
            squared_differences = (pointsInCluster - self.centers[clusterIndex]) ** 2

            # Summing up squared differences for all points in this cluster
            cluster_loss = np.sum(squared_differences)

            # Adding to the total loss
            total_loss += cluster_loss

        return total_loss


    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__ 
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.   
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """

        prevLoss = 0
        #update cluster assignment for each points
        step = 0
        done = False
        while (step < self.max_iters and not done):
            self.update_assignment()
            #Update the cluster centers based on the new assignments from Step 1
            self.update_centers()
            #Check to make sure there is no mean without a cluster, 
            #i.e. no cluster center without any points assigned to it.
            for clusterIndex in range(self.K):
                if len(self.points[self.assignments == clusterIndex]) == 0:
                    #pick a random point in the dataset to be the new center and 
                    #update your cluster assignment accordingly.

                    #boolean mask
                    pointIndices = np.arange(np.shape(self.points[0]))
                    noClusterPoints = np.setdiff1d(pointIndices, self.centers)
                    newCenter = noClusterPoints[np.randint(0, np.shape(noClusterPoints)[0])]
                    self.centers[clusterIndex] = self.points[newCenter]
                    self.update_assignment()
                    self.update_centers()
            #Calculate the loss
            loss = self.get_loss()

            #convergence criteria
            if (step == 0):
                prevLoss = self.get_loss()
            else:
                percentDiff = abs(loss - prevLoss)/prevLoss
                if (percentDiff < self.rel_tol):
                    done = True
                    break
            step += 1
            prevLoss = loss
        return self.centers, self.assignments, self.loss

def sum_squares(arr):
    return np.sum(np.square(arr), axis = 1, keepdims = True)

def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        sumX = np.sum(x**2, axis=1, keepdims=True)
        sumY = np.sum(y**2, axis=1, keepdims=True)

        dotProduct = x.dot (y.T)

        return np.sqrt(np.abs (sumX - 2*dotProduct + sumY.T))


import numpy as np

def rand_statistic(xGroundTruth, xPredicted):
    """
    Args:
        xPredicted : N x 1 numpy array, N = no. of test samples
        xGroundTruth: N x 1 numpy array, N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float
    """
    # Initialize counts for TP, TN, FP, FN
    TP, TN, FP, FN = 0, 0, 0, 0
    
    # Number of samples
    N = len(xGroundTruth)
    
    # Iterate through each pair of points to find TP, TN, FP, FN
    for i in range(N):
        for j in range(i+1, N):
            # Ground truth and predicted clusters for points i and j
            groundTruthI, groundTruthJ = xGroundTruth[i], xGroundTruth[j]
            predI, predJ = xPredicted[i], xPredicted[j]
            

            #classify type of result
            # True Positive
            if groundTruthI == groundTruthJ and predI == predJ:
                TP += 1

            # False Positive
            elif groundTruthI != groundTruthJ and predI == predJ:
                FP += 1

            # True Negative
            elif groundTruthI != groundTruthJ and predI != predJ:
                TN += 1
            
            # False Negative
            elif groundTruthI == groundTruthJ and predI != predJ:
                FN += 1
    
    return (TP + TN) / (TP + TN + FP + FN)
    
    