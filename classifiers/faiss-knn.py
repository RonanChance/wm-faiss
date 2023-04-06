import numpy as np
import faiss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class FaissKnn(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors classifier using the FAISS library for efficient similarity search.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    quantizer_ : faiss.IndexFlat, faiss.IndexIVFFlat, etc
        Quantizer used to discretize the feature space.

    index_ : faiss.IndexIVFFlat
        Inverted file index for similarity search.

    X_ : ndarray of shape (n_samples, n_features)
        Training input samples.

    y_ : ndarray of shape (n_samples,)
        Target values.

    n_features_in_ : int
        Number of features in the input data.

    Methods
    -------
    fit(X, y)
        Fit the model using X as training data and y as target values.

    predict(X)
        Predict the class labels for the provided data.

    score(X, y)
        Return the mean accuracy on the given test data and labels.

    get_params(deep=True)
        Get parameters for this estimator.

    set_params(**parameters)
        Set the parameters of this estimator.

    Notes
    -----
    This estimator uses the FAISS library to perform approximate nearest
    neighbors search.

    For more information on FAISS, see https://github.com/facebookresearch/faiss.

    For more information on k-nearest neighbors, see
    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm.
    """
    
    def __init__(self, n_neighbors=3, weights='uniform', p=2, n_centroids=1):
        """
        Initialize the KNN_FAISS instance.

        Parameters
        ----------
        n_neighbors : int, default=3
            Number of neighbors to use in k-nearest neighbors search.

        weights : str or callable, default='uniform'
            Weight function used in prediction. Possible values are 'uniform',
            'distance', 'distancesquared', or a callable object.

        p : int or str, default=2
            Distance metric to use for k-nearest neighbors search. Possible values
            are 1 for Manhattan, 2 for Euclidean, np.inf for Chebyshev, 'cosine'
            for cosine similarity, 'jaccard' for Jaccard distance, 'hamming' for
            Hamming distance, or a positive integer for Lp norm.

        n_centroids : int, default=1
            Number of centroids to use in IVF index.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.n_centroids = n_centroids

    def _get_quantizer(self, X):
        """
        Get a FAISS quantizer object for the specified distance metric.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        quantizer : faiss.Index
            The FAISS quantizer object for the specified distance metric.

        Raises
        ------
        ValueError
            If the specified distance metric is not recognized.
        """

        if self.p == 1:
            return faiss.IndexFlat(X.shape[1], faiss.METRIC_L1)
        elif self.p == 2:
            return faiss.IndexFlatL2(X.shape[1])
        elif self.p == np.inf:
            return faiss.IndexFlat(X.shape[1], faiss.METRIC_Linf)
        elif self.p > 0 and isinstance(self.p, int):
            return faiss.IndexFlat(X.shape[1], faiss.METRIC_Lp, self.p)
        elif self.p == 'cosine':
            return faiss.IndexFlatIP(X.shape[1], faiss.METRIC_)
        elif self.p == 'jaccard':
            return faiss.IndexFlatIP(X.shape[1], faiss.METRIC_Jaccard)
        elif self.p == 'hamming':
            return faiss.IndexBinaryFlat(X.shape[1])
        else:
            raise ValueError("Unrecognized distance: choose 1 for Manhattan, 2 for Euclidean, \
            'cosine' for cosine similarity, 'jaccard' for Jaccard distance, \
            'hamming' for Hamming distance, or a positive integer for Lp norm")

    def fit(self, X, y):
        """
        Fit the k-nearest neighbors classifier to the raining data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels) for the training samples.

        Returns
        -------
        self : KNN_FAISS
            The fitted k-nearest neighbors classifier.
        """
        X, y = check_X_y(X, y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X.shape[0] ({X.shape[0]}) is different from number of features in y.shape[0] ({y.shape[0]})")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("The input data is empty.")

        self.classes_ = unique_labels(y)

        self.quantizer_ = self._get_quantizer(X)
        # look into IndexFlat, IndexLSH, IndexPQ, IndexIVFPQ
        self.index_ = faiss.IndexIVFFlat(self.quantizer_, X.shape[1], self.n_centroids)
        self.index_.train(X.astype(np.float32))
        self.index_.add(X.astype(np.float32))

        self.X_, self.y_ = X, y
        self.n_features_in_ = X.shape[1]
        
        return self


    def _compute_weights(self, distances):
        """
        Compute the weights for each neighbor based on the specified weight scheme.

        Parameters
        ----------
        distances : array-like of shape (n_samples, n_neighbors)
            The distances to the nearest neighbors of each training sample.

        Returns
        -------
        weights_list : list of arrays of shape (n_neighbors, 2)
            The weights for each neighbor. Each entry is a tuple of weight and label.

        Raises
        -------
        ValueError : If the specified weight scheme is not recognized.

        Note
        ----
        the method assumes that self.weights has been set to a valid weight scheme
             or callable object during initialization.
        """
        if callable(self.weights):
            return self.weights(distances)
        elif self.weights == 'uniform':
            return [(1, y) for d, y in distances]
        elif self.weights == 'distance':
            return [(1/(d+1e-8), y) for d, y in distances]
        elif self.weights == 'distancesquared':
            return [(1/(d**2+1e-8), y) for d, y in distances]
        else:
            raise ValueError("Unrecognized weights: choose 'uniform', 'distance', 'distancesquared', or a callable object")


    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        results : numpy array, shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        ValueError 
            if the number of features is mismatched from features in fit
        """
        check_is_fitted(self)
        X = check_array(X)
        
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(f"Number of features in predict ({X.shape[1]}) is different from number of features in fit ({self.X_.shape[1]})")

        distances, indices = self.index_.search(X.astype(np.float32), self.n_neighbors)
        labels = self.y_[indices]
        weights_list = [self._compute_weights(zip(distances[i], labels[i])) for i in range(X.shape[0])]

        results = []
        for weights in weights_list:
            weights_by_class = defaultdict(list)
            for d, c in weights:
                weights_by_class[c].append(d)
            counts = [(sum(val), key) for key, val in weights_by_class.items()]
            results.append(max(counts, key=lambda x: x[0])[1])

        return np.asarray(results)


    def score(self, X, y):
        """
        Calculate accuracy of classifier on given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for `X`.

        Returns
        -------
        score : float
            Accuracy of predictions (Returns None if y is None).
        """
        if y is None:
            return None
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    # these two methods make it not dependent on sklearn BaseEstimator
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return the parameters of all sub-objects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"n_neighbors": self.n_neighbors, "weights": self.weights, 
                "p": self.p, "n_centroids": self.n_centroids}

    def set_params(self, **parameters):
        """
         Set the parameters of this estimator.

        Parameters
        ----------
        **parameters : dict
            A dictionary of parameter names mapped to their new values.

        Returns
        -------
        self : object
            Returns the estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self