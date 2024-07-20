import numpy as np
import sys
import numdifftools as nd
sys.path.append("./src")
from simple_ml import *
try:
    from simple_ml_ext import *
except:
    pass

##############################################################################
### TESTS/SUBMISSION CODE FOR parse_mnist()

def test_parse_mnist():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    np.testing.assert_allclose(np.linalg.norm(X[:1000]), 293.0717,
        err_msg="""If you failed this test but not the previous one,
        you are probably normalizing incorrectly. You should normalize
        w.r.t. the whole dataset, _not_ individual images.""", rtol=1e-6)
    np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_loss()

def test_softmax_loss():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)

    Z = np.zeros((y.shape[0], 10))
    np.testing.assert_allclose(softmax_loss(Z,y), 2.3025850)
    Z = np.random.randn(y.shape[0], 10)
    np.testing.assert_allclose(softmax_loss(Z,y), 2.7291998)

##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_regression_epoch()

def test_softmax_regression_epoch():
    # test numeical gradient
    np.random.seed(0)
    X = np.random.randn(50,5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    Theta = np.zeros((5,3), dtype=np.float32)
    dTheta = -nd.Gradient(lambda Th : softmax_loss(X@Th.reshape(5,3),y))(Theta)
    softmax_regression_epoch(X,y,Theta,lr=1.0,batch=50)
    np.testing.assert_allclose(dTheta.reshape(5,3), Theta, rtol=1e-4, atol=1e-4)


    # test multi-steps on MNIST
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
    softmax_regression_epoch(X[:100], y[:100], theta, lr=0.1, batch=10)
    np.testing.assert_allclose(np.linalg.norm(theta), 1.0947356, 
                               rtol=1e-5, atol=1e-5)