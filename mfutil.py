import numpy as np
import ctypes
import sys

mf = ctypes.CDLL(sys.argv[1])
c_float_p = ctypes.POINTER(ctypes.c_float)


class MFmodel(object):
    def __init__(self, k, m, n, b, fun, P, Q):
        self.k = k
        self.m = m
        self.n = n
        self.b = b
        self.fun = fun
        self.P = P
        self.Q = Q
        self.k_c = ctypes.c_int(self.k)
        self.m_c = ctypes.c_int(self.m)
        self.n_c = ctypes.c_int(self.n)
        self.b_c = ctypes.c_int(self.b)
        self.fun_c = ctypes.c_int(self.fun)
        temp_p = self.P.astype(np.float32)
        self.P_c = temp_p.ctypes.data_as(c_float_p)
        temp_q = self.Q.astype(np.float32)
        self.Q_c = temp_q.ctypes.data_as(c_float_p)

    def update_c_repr(self, k, m, n, b, fun, P, Q):
        self.k_c = ctypes.c_int(k)
        self.m_c = ctypes.c_int(m)
        self.n_c = ctypes.c_int(n)
        self.b_c = ctypes.c_int(b)
        self.fun_c = ctypes.c_int(fun)
        temp_p = P.astype(np.float32)
        self.P_c = temp_p.ctypes.data_as(c_float_p)
        temp_q = Q.astype(np.float32)
        self.Q_c = temp_q.ctypes.data_as(c_float_p)


def mf_train(X, path="testmodel.test.t"):
    ensure_width(X, 3)
    d = X.astype(np.float32)
    data_p = d.ctypes.data_as(c_float_p)
    nnx = ctypes.c_int(X.shape[0])
    c_str_path = ctypes.c_char_p(path)
    mf.train_interface(nnx, data_p, c_str_path)


def get_c_repr(x):
    """

    :param x: numpy array with shape (n, 3)
    :return: c_int size, c_int,
    """
    d = x.astype(np.float32)
    data_p = d.ctypes.data_as(c_float_p)
    return data_p


def ensure_width(x, width):
    if x.shape[1] != width:
        raise ValueError("must be sparse array of shape (n, {0})", width)


def mf_train_valid(X, V, path="testmodel.test.t"):
    ensure_width(X, 3)
    ensure_width(V, 3)
    nnx = ctypes.c_int(X.shape[0])
    nnx_valid = ctypes.c_int(V.shape[0])
    c_str_path = ctypes.c_char_p(path)
    train_p = get_c_repr(X)
    valid_p = get_c_repr(V)
    mf.train_valid_interface(nnx, nnx_valid, train_p, valid_p, c_str_path)


def generate_test_data(xs, ys, k):
    rx = np.random.random_integers(0, xs, k)
    ry = np.random.random_integers(0, ys, k)
    rv = np.random.rand(k)
    return np.vstack((rx, ry, rv)).transpose()


train = generate_test_data(1000, 1000, 10000)
valid = generate_test_data(1000, 1000, 2000)

mf_train_valid(train, valid)
