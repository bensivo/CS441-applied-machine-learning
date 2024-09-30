import numpy as np
from linear_predict import linear_predict

def linear_residuals(X,beta,Y):
    """
    Calculate residual vector using linear_predict function

        Parameters:
                X (np.array): A numpy array with the shape (N, d) where N is the number of data points and d is dimension
                Y (np.array): A numpy array with the shape (N,) where N is the number of data points
                beta (np.array): A numpy array of the shape (d+1,1) where d is the data dimension
                
        Returns:
                e (np.array): A numpy array with the shape (N, ) that represents the residual vector
    """
    
    assert X.ndim==2
    N = X.shape[0]
    d = X.shape[1]
    assert beta.shape == (d+1,1)
    assert Y.shape == (N,)

    predictions = linear_predict(X, beta)
    e = Y - predictions
    return e
    

if __name__ == '__main__':
    some_X = (np.arange(35).reshape(7,5) ** 13) % 20
    some_beta = 2.**(-np.arange(6).reshape(-1,1))
    some_Y = np.sum(some_X, axis=1)
    some_res = linear_residuals(some_X, some_beta, some_Y)
    assert np.array_equal(some_res.round(3), np.array([16.938, 35.844, 33.812, 59.281, 16.938, 39.719, 16.938]))
    print('All checks passed')
