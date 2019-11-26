import numpy as np
import ctypes
import os
import sys

if "LIBMF_OBJ" in os.environ:
    print("Using compiled .so file specified in LIBMF_OBJ:")
    compiled_src = os.environ["LIBMF_OBJ"]
elif len(sys.argv) > 1:
    print("Using 1st argument as .so file path:")
    compiled_src = sys.argv[1]
else:
    site_pkgs = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    print("Using file found in {}:".format(site_pkgs))
    possible_objs = os.listdir(site_pkgs)
    filtered = [f for f in possible_objs if f[-3:] == '.so' and 'libmf' in f]
    if len(filtered) > 0:
        compiled_src = os.path.join(site_pkgs, filtered[0])
    else:
        raise IOError("Compiled .so file not found. If you know where it is, " 
		      "specify the path in the LIBMF_OBJ environment variable")
 
print(compiled_src)
mf = ctypes.CDLL(compiled_src)
c_float_p = ctypes.POINTER(ctypes.c_float)

''' libmf enums '''

P_L2_MFR = 0
P_L1_MFR = 1
P_KL_MFR = 2
P_LR_MFC = 5
P_L2_MFC = 6
P_L1_MFC = 7
P_ROW_BPR_MFOC = 10
P_COL_BPR_MFOC = 11

RMSE = 0
MAE = 1
GKL = 2
LOGLOSS = 5
ACC = 6
ROW_MPR = 10
COL_MPR = 11
ROW_AUC = 12
COL_AUC = 13

''' libmf enums '''


def get_default_options():
    options = [
        ("fun", ctypes.c_int, P_L2_MFR),
        ("k", ctypes.c_int, 8),
        ("nr_threads", ctypes.c_int, 12),
        ("nr_bins", ctypes.c_int, 26),
        ("nr_iters", ctypes.c_int, 20),
        ("lambda_p1", ctypes.c_float, 0.04),
        ("lambda_p2", ctypes.c_float, 0.0),
        ("lambda_q1", ctypes.c_float, 0.04),
        ("lambda_q2", ctypes.c_float, 0.0),
        ("eta", ctypes.c_float, 0.1),
        ("do_nmf", ctypes.c_bool, False),
        ("quiet", ctypes.c_bool, False),
        ("copy_data", ctypes.c_bool, True)
    ]
    return options


class MFModel(ctypes.Structure):
    _fields_ = [("fun", ctypes.c_int),
                ("m", ctypes.c_int),
                ("n", ctypes.c_int),
                ("k", ctypes.c_int),
                ("b", ctypes.c_float),
                ("P", c_float_p),
                ("Q", c_float_p)]


class MFParam(ctypes.Structure):
    _fields_ = [(o[0], o[1]) for o in get_default_options()]

options_ptr = ctypes.POINTER(MFParam)


class MF(object):
    def __init__(self, *args, **kwargs):
        self.model = None
        self._options = MFParam()
        self.i = None
        self.j = None
        for kw in kwargs:
            if kw not in [i[0] for i in get_default_options()]:
                print("Unrecognized keyword argument '{0}={1}'".format(kw, kwargs[kw]))

        for item in get_default_options():
            if item[0] not in kwargs:
                value = item[2]
            else:
                value = kwargs[item[0]]

            if item[0] is "fun":
                self._options.fun = ctypes.c_int(value)
            elif item[0] is "k":
                self._options.k = ctypes.c_int(value)
            elif item[0] is "nr_threads":
                self._options.nr_threads = ctypes.c_int(value)
            elif item[0] is "nr_bins":
                self._options.nr_bins = ctypes.c_int(value)
            elif item[0] is "nr_iters":
                self._options.nr_iters = ctypes.c_int(value)
            elif item[0] is "lambda_p1":
                self._options.lambda_p1 = ctypes.c_float(value)
            elif item[0] is "lambda_p2":
                self._options.lambda_p2 = ctypes.c_float(value)
            elif item[0] is "lambda_q1":
                self._options.lambda_q1 = ctypes.c_float(value)
            elif item[0] is "lambda_q2":
                self._options.lambda_q2 = ctypes.c_float(value)
            elif item[0] is "eta":
                self._options.eta = ctypes.c_float(value)
            elif item[0] is "do_nmf":
                self._options.do_nmf = ctypes.c_bool(value)
            elif item[0] is "quiet":
                self._options.quiet = ctypes.c_bool(value)
            elif item[0] is "copy_data":
                self._options.copy_data = ctypes.c_bool(value)

    def predict(self, X):
        """
        assuming we have already run the fit method, predict the values at certain indices of the data matrix
        :param X: (n, 2) shaped numpy array
        :return: numpy array of length n
        """
        if self.model is None:
            return LookupError("no model data is saved, try running model.mf_fit(...) first")
        ensure_width(X, 2)
        nnx = X.shape[0]
        out = np.zeros(nnx)
        out = out.astype(np.float32)
        X = X.astype(np.float32)
        X_p = X.ctypes.data_as(c_float_p)
        nnx_p = ctypes.c_int(nnx)
        mf.pred_model_interface(nnx_p, X_p, ctypes.c_void_p(out.ctypes.data), ctypes.byref(self.model))
        return out

    def fit(self, X):
        """
        factorize the i x j data matrix X into (j, k) (k, i) sized matrices stored in MF.model
        :param X: (n, 3) shaped numpy array [known index and values of the data matrix]
        """
        ensure_width(X, 3)
        d = X.astype(np.float32)
        data_p = d.ctypes.data_as(c_float_p)
        nnx = ctypes.c_int(X.shape[0])
        mf.fit_interface.restype = ctypes.POINTER(MFModel)
        mf.fit_interface.argtypes = (ctypes.c_int, c_float_p, options_ptr)
        out = mf.fit_interface(nnx, data_p, self._options)
        self.model = out.contents

    def mf_cross_validation(self, X, folds=5):
        """
        :param X: (n, 3)
        :param folds: number of train / test splits
        :return: average score across all folds
        """
        ensure_width(X, 3)
        d = X.astype(np.float32)
        data_p = d.ctypes.data_as(c_float_p)
        nnx = ctypes.c_int(X.shape[0])
        mf.cross_valid_interface.restype = ctypes.c_double
        mf.cross_valid_interface.argtypes = (ctypes.c_int, c_float_p, options_ptr, ctypes.c_int)
        score = mf.cross_valid_interface(nnx, data_p, self._options, folds)
        return score

    def mf_train_test(self, X, V):
        ensure_width(X, 3)
        ensure_width(V, 3)
        nnx = ctypes.c_int(X.shape[0])
        nnx_valid = ctypes.c_int(V.shape[0])

        train_p = X.astype(np.float32)
        train_p = train_p.ctypes.data_as(c_float_p)

        test_p = V.astype(np.float32)
        test_p = test_p.ctypes.data_as(c_float_p)

        mf.train_valid_interface.restype = ctypes.POINTER(MFModel)
        mf.train_valid_interface.argtypes = (ctypes.c_int, ctypes.c_int, c_float_p, c_float_p, options_ptr)
        out = mf.train_valid_interface(nnx, nnx_valid, train_p, test_p, self._options)
        self.model = out.contents

    def q_factors(self):
        if self.model is None:
            return LookupError("no model data is saved, try running model.mf_fit(...) first")
        out = np.zeros(self.model.n * self.model.k)
        out = out.astype(np.float32)
        mf.get_Q(ctypes.c_void_p(out.ctypes.data), ctypes.byref(self.model))
        return out.reshape((self.model.n, self.model.k))

    def p_factors(self):
        if self.model is None:
            return LookupError("no model data is saved, try running model.mf_fit(...) first")
        out = np.zeros(self.model.m * self.model.k)
        out = out.astype(np.float32)
        mf.get_P(ctypes.c_void_p(out.ctypes.data), ctypes.byref(self.model))
        return out.reshape((self.model.m, self.model.k))


def ensure_width(x, width):
    if x.shape[1] != width:
        raise ValueError("must be sparse array of shape (n, {0})", width)


def generate_test_data(xs, ys, k, indices_only=False):
    rx = np.random.random_integers(0, xs, k)
    ry = np.random.random_integers(0, ys, k)
    rv = np.random.rand(k)
    return np.vstack((rx, ry, rv)).transpose().copy() if not indices_only else np.vstack((rx,ry)).transpose().copy()

