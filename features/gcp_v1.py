# -*- coding: utf-8 -*-

# Author: Sebastien Dubois 
#		  for ALFA Group MIT

from __future__ import print_function


import numpy as np
from scipy import linalg, optimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
# from sklearn.utils import check_random_state, check_arrays
from sklearn.gaussian_process import regression_models as regression
from sklearn.gaussian_process import correlation_models as correlation
from scipy.stats import norm
from scipy import stats

"""Utilities for input validation"""
# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
# License: BSD 3 clause

import warnings
import numbers

import scipy.sparse as sp

# from ..externals import six
# from .fixes import safe_copy


class DataConversionWarning(UserWarning):
    "A warning on implicit data conversions happening in the code"
    pass

warnings.simplefilter("always", DataConversionWarning)


class NonBLASDotWarning(UserWarning):
    "A warning on implicit dispatch to numpy.dot"
    pass


# Silenced by default to reduce verbosity. Turn on at runtime for
# performance profiling.
warnings.simplefilter('ignore', NonBLASDotWarning)


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""

    # First try an O(n) time, O(1) space solution for the common case that
    # there everything is finite; fall back to O(n) space np.isfinite to
    # prevent false positives from overflow in sum method.
    _assert_all_finite(X.data if sp.issparse(X) else X)


def safe_asarray(X, dtype=None, order=None, copy=False, force_all_finite=True):
    """Convert X to an array or CSC/CSR/COO sparse matrix.

    Prevents copying X when possible. Sparse matrices in CSR, CSC and COO
    formats are passed through. Other sparse formats are converted to CSR
    (somewhat arbitrarily).

    If a specific compressed sparse format is required, use atleast2d_or_cs{c,r}
    instead.
    """
    if sp.issparse(X):
        if not isinstance(X, (sp.coo_matrix, sp.csc_matrix, sp.csr_matrix)):
            X = X.tocsr()
        elif copy:
            X = X.copy()
        if force_all_finite:
            _assert_all_finite(X.data)
        # enforces dtype on data array (order should be kept the same).
        X.data = np.asarray(X.data, dtype=dtype)
    else:
        X = np.array(X, dtype=dtype, order=order, copy=copy)
        if force_all_finite:
            _assert_all_finite(X)
    return X


def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return safe_asarray(X, dtype=np.float64, copy=copy,
                            force_all_finite=force_all_finite)
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        return X.astype(np.float32 if X.dtype == np.int32 else np.float64)


def array2d(X, dtype=None, order=None, copy=False, force_all_finite=True):
    """Returns at least 2-d array with data from X"""
    if sp.issparse(X):
        raise TypeError('A sparse matrix was passed, but dense data '
                        'is required. Use X.toarray() to convert to dense.')
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    if force_all_finite:
        _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = safe_copy(X_2d)
    return X_2d


def _atleast2d_or_sparse(X, dtype, order, copy, sparse_class, convmethod,
                         check_same_type, force_all_finite):
    if sp.issparse(X):
        if check_same_type(X) and X.dtype == dtype:
            X = getattr(X, convmethod)(copy=copy)
        elif dtype is None or X.dtype == dtype:
            X = getattr(X, convmethod)()
        else:
            X = sparse_class(X, dtype=dtype)
        if force_all_finite:
            _assert_all_finite(X.data)
        X.data = np.array(X.data, copy=False, order=order)
    else:
        X = array2d(X, dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite)
    return X


def atleast2d_or_csc(X, dtype=None, order=None, copy=False,
                     force_all_finite=True):
    """Like numpy.atleast_2d, but converts sparse matrices to CSC format.

    Also, converts np.matrix to np.ndarray.
    """
    return _atleast2d_or_sparse(X, dtype, order, copy, sp.csc_matrix,
                                "tocsc", sp.isspmatrix_csc,
                                force_all_finite)


def atleast2d_or_csr(X, dtype=None, order=None, copy=False,
                     force_all_finite=True):
    """Like numpy.atleast_2d, but converts sparse matrices to CSR format

    Also, converts np.matrix to np.ndarray.
    """
    return _atleast2d_or_sparse(X, dtype, order, copy, sp.csr_matrix,
                                "tocsr", sp.isspmatrix_csr,
                                force_all_finite)


def _num_samples(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %r" % x)
    return x.shape[0] if hasattr(x, 'shape') else len(x)


def check_arrays(*arrays, **options):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.
    By default lists and tuples are converted to numpy arrays.

    It is possible to enforce certain properties, such as dtype, continguity
    and sparse matrix format (if a sparse matrix is passed).

    Converting lists to arrays can be disabled by setting ``allow_lists=True``.
    Lists can then contain arbitrary objects and are not checked for dtype,
    finiteness or anything else but length. Arrays are still checked
    and possibly converted.


    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays, unless allow_lists is specified.

    sparse_format : 'csr', 'csc' or 'dense', None by default
        If not None, any scipy.sparse matrix is converted to
        Compressed Sparse Rows or Compressed Sparse Columns representations.
        If 'dense', an error is raised when a sparse array is
        passed.

    copy : boolean, False by default
        If copy is True, ensure that returned arrays are copies of the original
        (if not already converted to another format earlier in the process).

    check_ccontiguous : boolean, False by default
        Check that the arrays are C contiguous

    dtype : a numpy dtype instance, None by default
        Enforce a specific dtype.

    allow_lists : bool
        Allow lists of arbitrary objects as input, just check their length.
        Disables

    allow_nans : boolean, False by default
        Allows nans in the arrays

    allow_nd : boolean, False by default
        Allows arrays of more than 2 dimensions.
    """
    sparse_format = options.pop('sparse_format', None)
    if sparse_format not in (None, 'csr', 'csc', 'dense'):
        raise ValueError('Unexpected sparse format: %r' % sparse_format)
    copy = options.pop('copy', False)
    check_ccontiguous = options.pop('check_ccontiguous', False)
    dtype = options.pop('dtype', None)
    allow_lists = options.pop('allow_lists', False)
    allow_nans = options.pop('allow_nans', False)
    allow_nd = options.pop('allow_nd', False)

    if options:
        raise TypeError("Unexpected keyword arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    n_samples = _num_samples(arrays[0])

    checked_arrays = []
    for array in arrays:
        array_orig = array
        if array is None:
            # special case: ignore optional y=None kwarg pattern
            checked_arrays.append(array)
            continue
        size = _num_samples(array)

        if size != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (size, n_samples))

        if not allow_lists or hasattr(array, "shape"):
            if sp.issparse(array):
                if sparse_format == 'csr':
                    array = array.tocsr()
                elif sparse_format == 'csc':
                    array = array.tocsc()
                elif sparse_format == 'dense':
                    raise TypeError('A sparse matrix was passed, but dense '
                                    'data is required. Use X.toarray() to '
                                    'convert to a dense numpy array.')
                if check_ccontiguous:
                    array.data = np.ascontiguousarray(array.data, dtype=dtype)
                elif hasattr(array, 'data'):
                    array.data = np.asarray(array.data, dtype=dtype)
                elif array.dtype != dtype:
                    array = array.astype(dtype)
                if not allow_nans:
                    if hasattr(array, 'data'):
                        _assert_all_finite(array.data)
                    else:
                        _assert_all_finite(array.values())
            else:
                if check_ccontiguous:
                    array = np.ascontiguousarray(array, dtype=dtype)
                else:
                    array = np.asarray(array, dtype=dtype)
                if not allow_nans:
                    _assert_all_finite(array)

            if not allow_nd and array.ndim >= 3:
                raise ValueError("Found array with dim %d. Expected <= 2" %
                                 array.ndim)

        if copy and array is array_orig:
            array = array.copy()
        checked_arrays.append(array)

    return checked_arrays


def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    Returns
    -------
    y : array

    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def warn_if_not_float(X, estimator='This algorithm'):
    """Warning utility function to check that data type is floating point.

    Returns True if a warning was raised (i.e. the input is not float) and
    False otherwise, for easier input validation.
    """
    if not isinstance(estimator, six.string_types):
        estimator = estimator.__class__.__name__
    if X.dtype.kind != 'f':
        warnings.warn("%s assumes floating point values as input, "
                      "got %s" % (estimator, X.dtype))
        return True
    return False


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)



MACHINE_EPSILON = np.finfo(np.double).eps

def find_bounds(f, y):
    x = 1
    while(f(x) < y):
        x = x * 2
    lo = -100 
    if (x ==1):
        lo = -100
    else:
        lo = x/2
    return lo, x
	
def binary_search(f, y, lo, hi, delta):
	while lo <= hi:
		x = (lo + hi) / 2
		#print(x)
		if f(x) < y:
			lo = x + delta
		elif f(x) > y:
			hi = x - delta
		else:
			return x 
	if (f(hi) - y < y - f(lo)):
		return hi
	else:
		return lo


def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = array2d(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij
	
	
def sq_exponential(theta,d):
	return np.exp( - theta[0] * np.sum(d ** 2, axis=1)  )


def exponential_periodic(theta,d):
	t0 = theta[0]
	t1 = theta[1]
	t2 = theta[2]
	t3 = theta[3]
	t4 = theta[4]
	t5 = theta[5]
	t6 = theta[6]
	t7 = theta[7]
	
	good_cond =  (t0 > 0) and (t1 > 0) and (t2 > 0) and (t6 > 0) 
	c = t0 + t1 + t2
	if(good_cond):
		c1 = t0 * np.exp( - t3 * np.sum(d ** 2, axis=1)  )
		c2 = t1 * np.exp( - (np.sum(d**2,axis=1)/(2.*t4*t4)) - 2*(np.sin(3.14 * np.sum( d, axis=1)) /t5)**2  )
		c3 = t2 * ( (np.prod(1+ (d/t7)**2 ) )** (-t6))
		return ((c1+c2+c3)/c)
	else:
		return np.asarray([0.])


class GaussianCopulaProcess(BaseEstimator, RegressorMixin):
	"""The Gaussian Copula Process model class.

	Parameters
	----------
	regr : string or callable, optional
		A regression function returning an array of outputs of the linear
		regression functional basis. The number of observations n_samples
		should be greater than the size p of this basis.
		Default assumes a simple constant regression trend.
		Available built-in regression models are::

			'constant', 'linear', 'quadratic'
		
	corr : string or callable, optional
		A stationary autocorrelation function returning the autocorrelation
		between two points x and x'.
		Default assumes a squared-exponential autocorrelation model.
		Built-in correlation models are::
			'absolute_exponential', 'squared_exponential',
			'generalized_exponential', 'cubic', 'linear'

	verbose : boolean, optional
		A boolean specifying the verbose level.
		Default is verbose = False.

	theta0 : double array_like, optional
		An array with shape (n_features, ) or (1, ).
		The parameters in the autocorrelation model.
		If thetaL and thetaU are also specified, theta0 is considered as
		the starting point for the maximum likelihood estimation of the
		best set of parameters.
		Default assumes isotropic autocorrelation model with theta0 = 1e-1.

	thetaL : double array_like, optional
		An array with shape matching theta0's.
		Lower bound on the autocorrelation parameters for maximum
		likelihood estimation.
		Default is None, so that it skips maximum likelihood estimation and
		it uses theta0.

	thetaU : double array_like, optional
		An array with shape matching theta0's.
		Upper bound on the autocorrelation parameters for maximum
		likelihood estimation.
		Default is None, so that it skips maximum likelihood estimation and
		it uses theta0.

	normalize : boolean, optional
		Input X and observations y are centered and reduced wrt
		means and standard deviations estimated from the n_samples
		observations provided.
		Default is normalize = True so that data is normalized to ease
		maximum likelihood estimation.

	nugget : double or ndarray, optional
		Introduce a nugget effect to allow smooth predictions from noisy
		data.  If nugget is an ndarray, it must be the same length as the
		number of data points used for the fit.
		The nugget is added to the diagonal of the assumed training covariance;
		in this way it acts as a Tikhonov regularization in the problem.  In
		the special case of the squared exponential correlation function, the
		nugget mathematically represents the variance of the input values.
		Default assumes a nugget close to machine precision for the sake of
		robustness (nugget = 10. * MACHINE_EPSILON).

	random_start : int, optional
		The number of times the Maximum Likelihood Estimation should be
		performed from a random starting point.
		The first MLE always uses the specified starting point (theta0),
		the next starting points are picked at random according to an
		exponential distribution (log-uniform on [thetaL, thetaU]).
		Default does not use random starting point (random_start = 1).

	random_state: integer or numpy.RandomState, optional
		The generator used to shuffle the sequence of coordinates of theta in
		the Welch optimizer. If an integer is given, it fixes the seed.
		Defaults to the global numpy random number generator.


	Attributes
	----------
	`theta_`: array
		Specified theta OR the best set of autocorrelation parameters (the \
		sought maximizer of the reduced likelihood function).

	`reduced_likelihood_function_value_`: array
		The optimal reduced likelihood function value.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.gaussian_process import GaussianProcess
	>>> X = np.array([[1., 3., 5., 6., 7., 8.]]).T
	>>> y = (X * np.sin(X)).ravel()
	>>> gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
	>>> gp.fit(X, y)                                      # doctest: +ELLIPSIS
	GaussianProcess(beta0=None...
			...

	Notes
	-----
	The presentation implementation is based on a translation of the DACE
	Matlab toolbox, see reference [NLNS2002]_.

	References
	----------
	On Gaussian processes :
	.. [NLNS2002] `H.B. Nielsen, S.N. Lophaven, H. B. Nielsen and J.
		Sondergaard.  DACE - A MATLAB Kriging Toolbox.` (2002)
		http://www2.imm.dtu.dk/~hbn/dace/dace.pdf

	.. [WBSWM1992] `W.J. Welch, R.J. Buck, J. Sacks, H.P. Wynn, T.J. Mitchell,
		and M.D.  Morris (1992). Screening, predicting, and computer
		experiments.  Technometrics, 34(1) 15--25.`
		http://www.jstor.org/pss/1269548
	On Gaussian Copula processes :
	.. Wilson, A. and Ghahramani, Z. Copula processes. In Advances in NIPS 23,
		pp. 2460-2468, 2010
	"""

	_regression_types = {
		'constant': regression.constant,
		'linear': regression.linear,
		'quadratic': regression.quadratic}

	_correlation_types = {
		'absolute_exponential': correlation.absolute_exponential,
		'squared_exponential': correlation.squared_exponential,
		'generalized_exponential': correlation.generalized_exponential,
		'cubic': correlation.cubic,
		'linear': correlation.linear}

	_optimizer_types = [
		'fmin_cobyla',
		'Welch']

	def __init__(self, regr='constant', 
				 corr='exponential_periodic',
				 verbose=False, 
				 theta=np.asarray([0.4,0.3,0.3,.05,0.1,1.,2.,.1]),
				 thetaL=np.asarray([.1,.1,.1,.001,0.01,0.01,0.1,0.001]),
				 thetaU=np.asarray([.9,.5,.2,.5,0.5,10.,10.,1.]), 
				 try_optimize=True,
				 random_start=10, 
				 normalize=True,
				 nugget=10. * MACHINE_EPSILON,
				 random_state=None):
 
		self.regr = regr
		if (corr == 'squared_exponential'):
			self.corr = sq_exponential
		else:
			self.corr = exponential_periodic
		self.beta0 = None
		self.storage_mode = 'full'
		self.verbose = verbose
		self.theta = theta
		self.thetaL = thetaL
		self.thetaU = thetaU
		self.normalize = normalize
		self.nugget = nugget
		self.optimizer = 'fmin_cobyla'
		self.random_start = random_start
		self.random_state = random_state
		self.try_optimize = try_optimize

		
	def mapping(self,t):
		temp = self.density.integrate_box_1d(self.low_bound, t)
		return [norm.ppf(temp)]
	
		
	def mapping_inv(self,t):
		def map(t):
			return self.mapping(t)
		delta=0.0001/1024.
		lo, hi = find_bounds(map, t)
		res = binary_search(map, t, lo, hi, delta)				
		return [res]

	
	def update_copula_params(self):
		size = self.raw_y.shape[0]
		y = [ self.mapping(self.raw_y[i]) for i in range(size)]
		y = np.asarray(y)
		
		# Normalize data or don't
		if True:
			y_mean = np.mean(y, axis=0)
			y_std = np.std(y, axis=0)
			y_std[y_std == 0.] = 1.
			# center and scale Y if necessary
			y = (y - y_mean) / y_std
		else:
			y_mean = np.zeros(1)
			y_std = np.ones(1)
		
		# Calculate matrix of distances D between samples
		D, ij = l1_cross_distances(self.X)
		if (np.min(np.sum(D, axis=1)) == 0.
				and self.corr != correlation.pure_nugget):
			raise Exception("Multiple input features cannot have the same"
							" target value.")

		n_samples = self.X.shape[0]
		# Regression matrix and parameters
		F = self.regr(self.X)
		n_samples_F = F.shape[0]
		if F.ndim > 1:
			p = F.shape[1]
		else:
			p = 1
		if n_samples_F != n_samples:
			raise Exception("Number of rows in F and X do not match. Most "
							"likely something is going wrong with the "
							"regression model.")
		if p > n_samples_F:
			raise Exception(("Ordinary least squares problem is undetermined "
							 "n_samples=%d must be greater than the "
							 "regression model size p=%d.") % (n_samples, p))

		self.D = D
		self.ij = ij
		self.F = F
		self.y = y
		self.y_mean, self.y_std = y_mean, y_std
			
			
	def fit(self, X, y):
		"""
		The Gaussian Copula Process model fitting method.

		Parameters
		----------
		X : double array_like
			An array with shape (n_samples, n_features) with the input at which
			observations were made.

		y : double array_like
			An array with shape (n_samples, ) or shape (n_samples, n_targets)
			with the observations of the output to be predicted.

		Returns
		-------
		gp : self
			A fitted Gaussian Process model object awaiting data to perform
			predictions.
		"""
		# Run input checks
		self._check_params()
		X = array2d(X)
		y = np.asarray(y)
		self.y_ndim_ = y.ndim
		if y.ndim == 1:
			y = y[:, np.newaxis]
			
		self.random_state = check_random_state(self.random_state)
		self.raw_y = y
		self.low_bound = np.min([-500., 5. * np.min(y)])
		self.density = stats.gaussian_kde(self.raw_y[:,0])
		

		X, y = check_arrays(X, y)

		# Check shapes of DOE & observations
		n_samples, n_features = X.shape
		_, n_targets = y.shape

		# Run input checks
		self._check_params(n_samples)

		# Normalize data or don't
		if self.normalize:
			X_mean = np.mean(X, axis=0)
			X_std = np.std(X, axis=0)
			X_std[X_std == 0.] = 1.
			# center and scale X if necessary
			X = (X - X_mean) / X_std
		else:
			X_mean = np.zeros(1)
			X_std = np.ones(1)

		# Set attributes
		self.X = X
		self.X_mean, self.X_std = X_mean, X_std

		self.update_copula_params()
		
		if self.try_optimize:
		    # Maximum Likelihood Estimation of the parameters
			if self.verbose:
				print("Performing Maximum Likelihood Estimation of the "
					  "autocorrelation parameters...")
			self.theta, self.reduced_likelihood_function_value_, par = \
				self._arg_max_reduced_likelihood_function()
			if np.isinf(self.reduced_likelihood_function_value_):
				raise Exception("Bad parameter region. "
								"Try increasing upper bound")
		else:
			# Given parameters
			if self.verbose:
				print("Given autocorrelation parameters. "
					  "Computing Gaussian Process model parameters...")
			self.reduced_likelihood_function_value_, par = \
				self.reduced_likelihood_function()
			if np.isinf(self.reduced_likelihood_function_value_):
				raise Exception("Bad point. Try increasing theta0.")

		self.beta = par['beta']
		self.gamma = par['gamma']
		self.sigma2 = par['sigma2']
		self.C = par['C']
		self.Ft = par['Ft']
		self.G = par['G']

		return self

	def predict(self, X, eval_MSE=False, batch_size=None):
		"""
		This function evaluates the Gaussian Process model at x.

		Parameters
		----------
		X : array_like
			An array with shape (n_eval, n_features) giving the point(s) at
			which the prediction(s) should be made.

		eval_MSE : boolean, optional
			A boolean specifying whether the Mean Squared Error should be
			evaluated or not.
			Default assumes evalMSE = False and evaluates only the BLUP (mean
			prediction).

		batch_size : integer, optional
			An integer giving the maximum number of points that can be
			evaluated simultaneously (depending on the available memory).
			Default is None so that all given points are evaluated at the same
			time.

		Returns
		-------
		y : array_like, shape (n_samples, ) or (n_samples, n_targets)
			An array with shape (n_eval, ) if the Gaussian Process was trained
			on an array of shape (n_samples, ) or an array with shape
			(n_eval, n_targets) if the Gaussian Process was trained on an array
			of shape (n_samples, n_targets) with the Best Linear Unbiased
			Prediction at x.

		MSE : array_like, optional (if eval_MSE == True)
			An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
			with the Mean Squared Error at x.
		"""

		# Check input shapes
		X = array2d(X)
		n_eval, _ = X.shape
		n_samples, n_features = self.X.shape
		n_samples_y, n_targets = self.y.shape

		# Run input checks
		self._check_params(n_samples)

		if X.shape[1] != n_features:
			raise ValueError(("The number of features in X (X.shape[1] = %d) "
							  "should match the number of features used "
							  "for fit() "
							  "which is %d.") % (X.shape[1], n_features))

		# No memory management
		# (evaluates all given points in a single batch run)

		# Normalize input
		if self.normalize:
			X = (X - self.X_mean) / self.X_std

		# Initialize output
		y = np.zeros(n_eval)
		if eval_MSE:
			MSE = np.zeros(n_eval)

		# Get pairwise componentwise L1-distances to the input training set
		dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
		# Get regression function and correlation
		f = self.regr(X)
		r = self.corr(self.theta, dx).reshape(n_eval, n_samples)

		# Scaled predictor
		y_ = np.dot(f, self.beta) + np.dot(r, self.gamma)

		# Predictor
		y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

		# transform the gaussian y to the real y
		size = y.shape[0]
		real_y = [ self.mapping_inv(y[i][0]) for i in range(size)]
		real_y = np.asarray(real_y)
		y = real_y.reshape(n_eval, n_targets)
		
		if self.y_ndim_ == 1:
			y = y.ravel()
			
					# Mean Squared Error
		if eval_MSE:
			C = self.C
			if C is None:
				# Light storage mode (need to recompute C, F, Ft and G)
				if self.verbose:
					print("This GaussianProcess used 'light' storage mode "
						  "at instantiation. Need to recompute "
						  "autocorrelation matrix...")
				reduced_likelihood_function_value, par = \
					self.reduced_likelihood_function()
				self.C = par['C']
				self.Ft = par['Ft']
				self.G = par['G']

			rt = linalg.solve_triangular(self.C, r.T, lower=True)

			if self.beta0 is None:
				# Universal Kriging
				u = linalg.solve_triangular(self.G.T,
											np.dot(self.Ft.T, rt) - f.T)
			else:
				# Ordinary Kriging
				u = np.zeros((n_targets, n_eval))

			MSE = np.dot(self.sigma2.reshape(n_targets, 1),
						 (1. - (rt ** 2.).sum(axis=0)
						  + (u ** 2.).sum(axis=0))[np.newaxis, :])
			MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

			# Mean Squared Error might be slightly negative depending on
			# machine precision: force to zero!
			MSE[MSE < 0.] = 0.

			if self.y_ndim_ == 1:
				MSE = MSE.ravel()

			return y, MSE

		else:
			return y


	def reduced_likelihood_function(self, theta=None,verb=False):
		"""
		This function determines the BLUP parameters and evaluates the reduced
		likelihood function for the given autocorrelation parameters theta.

		Maximizing this function wrt the autocorrelation parameters theta is
		equivalent to maximizing the likelihood of the assumed joint Gaussian
		distribution of the observations y evaluated onto the design of
		experiments X.

		Parameters
		----------
		theta : array_like, optional
			An array containing the autocorrelation parameters at which the
			Gaussian Process model parameters should be determined.
			Default uses the built-in autocorrelation parameters
			(ie ``theta = self.theta_``).

		Returns
		-------
		reduced_likelihood_function_value : double
			The value of the reduced likelihood function associated to the
			given autocorrelation parameters theta.

		par : dict
			A dictionary containing the requested Gaussian Process model
			parameters:

				sigma2
						Gaussian Process variance.
				beta
						Generalized least-squares regression weights for
						Universal Kriging or given beta0 for Ordinary
						Kriging.
				gamma
						Gaussian Process weights.
				C
						Cholesky decomposition of the correlation matrix [R].
				Ft
						Solution of the linear equation system : [R] x Ft = F
				G
						QR decomposition of the matrix Ft.
		"""
		
		if theta is None:
			# Use built-in autocorrelation parameters
			theta = self.theta

		# Initialize output
		reduced_likelihood_function_value = - np.inf
		par = {}

		# Retrieve data
		n_samples = self.X.shape[0]
		D = self.D
		ij = self.ij
		F = self.F

		if D is None:
			# Light storage mode (need to recompute D, ij and F)
			D, ij = l1_cross_distances(self.X)
			if (np.min(np.sum(D, axis=1)) == 0.
					and self.corr != correlation.pure_nugget):
				raise Exception("Multiple X are not allowed")
			F = self.regr(self.X)

		# Set up R
		r = self.corr(theta, D)
		R = np.eye(n_samples) * (1. + self.nugget)
		R[ij[:, 0], ij[:, 1]] = r
		R[ij[:, 1], ij[:, 0]] = r

		# Cholesky decomposition of R
		try:
			C = linalg.cholesky(R, lower=True)
		except linalg.LinAlgError:
			return reduced_likelihood_function_value, par

		# Get generalized least squares solution
		Ft = linalg.solve_triangular(C, F, lower=True)
		try:
			Q, G = linalg.qr(Ft, econ=True)
		except:
			#/usr/lib/python2.6/dist-packages/scipy/linalg/decomp.py:1177:
			# DeprecationWarning: qr econ argument will be removed after scipy
			# 0.7. The economy transform will then be available through the
			# mode='economic' argument.
			Q, G = linalg.qr(Ft, mode='economic')
			pass

		sv = linalg.svd(G, compute_uv=False)
		rcondG = sv[-1] / sv[0]
		if rcondG < 1e-10:
			# Check F
			sv = linalg.svd(F, compute_uv=False)
			condF = sv[0] / sv[-1]
			if condF > 1e15:
				raise Exception("F is too ill conditioned. Poor combination "
								"of regression model and observations.")
			else:
				# Ft is too ill conditioned, get out (try different theta)
				return reduced_likelihood_function_value, par

		Yt = linalg.solve_triangular(C, self.y, lower=True)
		if self.beta0 is None:
			# Universal Kriging
			beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
		else:
			# Ordinary Kriging
			beta = np.array(self.beta0)

		rho = Yt - np.dot(Ft, beta)
		sigma2 = (rho ** 2.).sum(axis=0) / n_samples
		# The determinant of R is equal to the squared product of the diagonal
		# elements of its Cholesky decomposition C
		detR = (np.diag(C) ** (2. / n_samples)).prod()

		# Compute/Organize output
		reduced_likelihood_function_value = - sigma2.sum() * detR
		par['sigma2'] = sigma2 * self.y_std ** 2.
		par['beta'] = beta
		par['gamma'] = linalg.solve_triangular(C.T, rho)
		par['C'] = C
		par['Ft'] = Ft
		par['G'] = G

		return reduced_likelihood_function_value, par

	def _arg_max_reduced_likelihood_function(self):
		"""
		This function estimates the autocorrelation parameters theta as the
		maximizer of the reduced likelihood function.
		(Minimization of the opposite reduced likelihood function is used for
		convenience)

		Parameters
		----------
		self : All parameters are stored in the Gaussian Process model object.

		Returns
		-------
		optimal_theta : array_like
			The best set of autocorrelation parameters (the sought maximizer of
			the reduced likelihood function).

		optimal_reduced_likelihood_function_value : double
			The optimal reduced likelihood function value.

		optimal_par : dict
			The BLUP parameters associated to thetaOpt.
		"""
		

		# Initialize output
		best_optimal_theta = []
		best_optimal_rlf_value = []
		best_optimal_par = []

		if self.verbose:
			print("The chosen optimizer is: " + str(self.optimizer))
			if self.random_start > 1:
				print(str(self.random_start) + " random starts are required.")

		percent_completed = 0.

		if self.optimizer == 'fmin_cobyla':

			def minus_reduced_likelihood_function(x):
				#print(x)
				if( ((1. - ((10. ** x[0]) + (10. ** x[1]) + (10. ** x[2]))) <0)):
					return 999999999.
				else:
					return - self.reduced_likelihood_function(
						theta=10. ** x,
						#copula_par=10 ** x[T:],
						verb=False)[0]
						
			constraints = []
			conL = np.log10(self.thetaL)
			conU = np.log10(self.thetaU)
			def constrL(x):
				x - conL
			def constrU(x):
				conU - x
			def kernel_coef(x):
				(1. - ((10. ** x[0]) + (10. ** x[1]) + (10. ** x[2])))

			constraints.append(constrL)
			constraints.append(constrU)
			
			if(self.theta.shape[0] > 1):
				constraints.append(kernel_coef)
			
			k=0
			k2 = 0
			while( (k < self.random_start) and (k2 < 500)):

				if k == 0:
					# Use specified starting point as first guess
					theta0 = self.theta

				else:
					# Generate a random starting point log10-uniformly
					# distributed between bounds
					log10theta0 = np.log10(self.thetaL) \
						+ self.random_state.rand(self.theta.size).reshape(
							self.theta.shape) * np.log10(self.thetaU
														  / self.thetaL)
					theta0 = 10. ** log10theta0
					
				# Run Cobyla
				try:
					params= np.log10(theta0)
					log10_opt = \
						optimize.fmin_cobyla(minus_reduced_likelihood_function,
											 params,
											 constraints,
											 iprint=0)
					opt_minus_rlf = minus_reduced_likelihood_function(log10_opt)
					#print(opt_minus_rlf)
					log10_optimal_theta = log10_opt
					
				except ValueError as ve:
					print("Optimization failed. Try increasing the ``nugget``")
					raise ve
				
				if(opt_minus_rlf != 999999999. ):
					k2=0
					optimal_theta = 10. ** log10_optimal_theta
					# print(optimal_theta)
					optimal_rlf_value, optimal_par = self.reduced_likelihood_function(theta=optimal_theta)

					# Compare the new optimizer to the best previous one
					if k > 0:
						if optimal_rlf_value > best_optimal_rlf_value:
							best_optimal_rlf_value = optimal_rlf_value
							best_optimal_par = optimal_par
							best_optimal_theta = optimal_theta
					else:
						best_optimal_rlf_value = optimal_rlf_value
						best_optimal_par = optimal_par
						best_optimal_theta = optimal_theta
					if self.verbose and self.random_start > 1:
						if (20 * k) / self.random_start > percent_completed:
							percent_completed = (20 * k) / self.random_start
							print("%s completed" % (5 * percent_completed))
					
					k += 1
					
				else:
					k2 += 1
					if(k2 == 500):
						print('MLE failed...')
						best_optimal_theta = self.theta
						best_optimal_rlf_value, best_optimal_par = self.reduced_likelihood_function(theta=best_optimal_theta)
									
			optimal_rlf_value = best_optimal_rlf_value
			optimal_par = best_optimal_par
			optimal_theta = best_optimal_theta
			
		else:
			raise NotImplementedError("This optimizer ('%s') is not "
									  "implemented yet. Please contribute!"
									  % self.optimizer)

		return optimal_theta, optimal_rlf_value, optimal_par


	def _check_params(self, n_samples=None):

		# Check regression model
		if not callable(self.regr):
			if self.regr in self._regression_types:
				self.regr = self._regression_types[self.regr]
			else:
				raise ValueError("regr should be one of %s or callable, "
								 "%s was given."
								 % (self._regression_types.keys(), self.regr))

		# Check correlation model
		if not callable(self.corr):
			if self.corr in self._correlation_types:
				self.corr = self._correlation_types[self.corr]
			else:
				raise ValueError("corr should be one of %s or callable, "
								 "%s was given."
								 % (self._correlation_types.keys(), self.corr))


		# Force verbose type to bool
		self.verbose = bool(self.verbose)

		# Force normalize type to bool
		self.normalize = bool(self.normalize)

		# Check nugget value
		self.nugget = np.asarray(self.nugget)
		if np.any(self.nugget) < 0.:
			raise ValueError("nugget must be positive or zero.")
		if (n_samples is not None
				and self.nugget.shape not in [(), (n_samples,)]):
			raise ValueError("nugget must be either a scalar "
							 "or array of length n_samples.")

		# Check optimizer
		if not self.optimizer in self._optimizer_types:
			raise ValueError("optimizer should be one of %s"
							 % self._optimizer_types)

		# Force random_start type to int
		self.random_start = int(self.random_start)
