if full_matrix is False:
            N, D = self.points.shape  # Number of data points
            K = len(pi)  # Number of clusters
            
            # holds log liklihood
            matrix = np.zeros((N, K))
            
            for k in range(K):
                values = self.normalPDF(self.points, mu[k], sigma[k])
                
                # Compute the log-likelihood for each component for each data point
                matrix[:, k] = np.log(pi[k]) + np.log(values + LOG_CONST)
            
            return matrix