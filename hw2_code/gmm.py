import numpy as np
from tqdm import tqdm


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = False # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        maxVal = np.max(logit, axis=1, keepdims=True)
        logit = logit - maxVal #subtract away largest
        expLogit = np.exp(logit) 
        sum = np.sum(expLogit, axis=1, keepdims=True) #sum accross rows
        return expLogit / sum
    
    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        maxVal = np.max(logit, axis=1, keepdims=True)  # Max of the logits, not the exponentiated values
        noMax = logit - maxVal
        summedExp = np.sum(np.exp(noMax), axis=1, keepdims=True)

        return maxVal + np.log(summedExp)

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        SMALL_CONST = 1e-9
        variances = np.diagonal(sigma_i) + SMALL_CONST

        # Compute the normalization constant
        determinant = np.prod(variances)
        constant = 1 / (np.sqrt((2 * np.pi) ** points.shape[1] * determinant))

        # Compute the deviations and the exponential term
        deviations = points - mu_i
        squared_deviations = np.square(deviations)
        exp_term = np.exp(-0.5 * squared_deviations / variances)

        # Compute the PDF
        return constant * np.prod(exp_term, axis=1)
            
    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """ 

        raise NotImplementedError


    def create_pi(self):
        """
        Initialize the prior probabilities 
        Args:
        Return:
        pi: numpy array of length K, prior
        """

        pi = np.full(self.K, 1 / self.K)
        
        return pi

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        mu = np.zeros((self.K, self.D))
        for index in range(self.K):
            mu[index] = self.points[np.random.choice(self.N)]
        return mu
    
    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the 
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        sigma = np.array([np.eye(self.D)] * self.K)
        
        return sigma
    
    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed
        # Initialize the mixing coefficients (priors) to be uniform
        pi = self.create_pi()
        
        # Initialize the means randomly.
        # Here we assume data points are in a unit cube; adjust the range as needed.
        mu = self.create_mu()
        
        # Initialize the diagonal covariance matrices.
        # We'll set them to identity matrices for each component.
        sigma = self.create_sigma()

        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        if full_matrix == True:
            return None
        if full_matrix is False:
            N = self.N  # Number of data points
            K = self.K  # Number of clusters
            
            # holds log liklihood
            matrix = np.zeros((N, K))
            
            for k in range(K):
                values = self.normalPDF(self.points, mu[k], sigma[k])
                
                # Compute the log-likelihood for each component for each data point
                matrix[:, k] = np.log(pi[k] + LOG_CONST) + np.log(values + LOG_CONST)
            
            return matrix



    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
       

        # === undergraduate implementation
        if full_matrix is False:
            # Compute log-likelihood 
            matrix = self._ll_joint(pi, mu, sigma, full_matrix = False)
            
            # call softmax
            return self.softmax(matrix)
            

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """

        # === undergraduate implementation
        if full_matrix is False:
            

            N, D = self.points.shape
            K = gamma.shape[1]
            
            # Calculate N_k for each component
            point = np.sum(gamma, axis=0)
            
            # Update pi
            pi_new = point / N
            
            # Update mu
            mu_new = np.dot(gamma.T, self.points) / point[:, np.newaxis]
            
            # Update sigma
            sigma_new = np.zeros((K, D, D))
            
            for k in range(K):
                for n in range(N):
                    diff = (self.points[n] - mu_new[k]).reshape(-1, 1)
                    sigma_new[k] += gamma[n, k] * np.dot(diff, diff.T)
                
                sigma_new[k] /= point[k]
                
                if not full_matrix:
                    sigma_new[k] = np.diag(np.diag(sigma_new[k]))
        return pi_new, mu_new, sigma_new

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

