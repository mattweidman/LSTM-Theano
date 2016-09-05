import numpy as np
import theano
import theano.tensor as T
from theano import function

from LSTM_layer import LSTM_layer

def test_LSTM_layer():
    n_in = 10
    n_out = 15
    n_ex = 20
    X_val = np.random.randn(n_ex, n_in)
    s_prev_val = np.random.randn(n_ex, n_out)
    h_prev_val = np.random.randn(n_ex, n_out)

    layer = LSTM_layer(n_in, n_out)
    X = T.dmatrix('X')
    s_prev = T.dmatrix('s_prev')
    h_prev = T.dmatrix('h_prev')
    s, h = layer.output(X, s_prev, h_prev)
    f = function([X, s_prev, h_prev], [s, h])
    print(f(X_val, s_prev_val, h_prev_val))

def test_layer_chars():

    alph_in = ['a', 'b', 'c', 'd', 'e']
    alph_out = ['a', 'b', 'c']
    char_to_ind = dict((i, c) for c, i in enumerate(alph_in))

    def c2vec(c, veclen):
        ind = char_to_ind[c]
        ans = np.zeros((veclen))
        ans[ind] = 1.0
        return ans

    def mat2str(mat):
        s = ""
        for row in mat:
            ind = np.argmax(row)
            s += alph_in[ind]
        return s

    # construct values for input matrices
    n_in = len(alph_in)
    n_out = len(alph_out)
    X_np = np.array([c2vec('a', n_in), c2vec('b', n_in), c2vec('c', n_in),
        c2vec('d', n_in), c2vec('e', n_in)])
    s_prev_np = np.zeros((len(X_np), len(alph_out)))
    h_prev_np = np.zeros((len(X_np), len(alph_out)))
    Y_np = np.array([c2vec('b', n_out), c2vec('c', n_out), c2vec('a', n_out),
        c2vec('b', n_out), c2vec('c', n_out)])
    num_ex = X_np.shape[0]
    print(mat2str(X_np))
    print(mat2str(Y_np))

    # construct theano input matrices
    X = T.dmatrix('X')
    s_prev = T.dmatrix('s_prev')
    h_prev = T.dmatrix('h_prev')
    Y = T.dmatrix('Y')

    # construct LSTM and output
    layer = LSTM_layer(len(alph_in), len(alph_out))
    s, h = layer.output(X, s_prev, h_prev)

    # calculate gradients and training model
    cost = (1/num_ex * (h-Y)**2).sum()
    gradient = T.grad(cost, layer.get_param_list())
    updates = [(param, param-gparam) for param, gparam in
        zip(layer.get_param_list(), gradient)]
    train_model = function([X, s_prev, h_prev, Y], cost, updates=updates)

    # train the model
    num_epochs = 1000
    for i in range(num_epochs):
        c = train_model(X_np, s_prev_np, h_prev_np, Y_np)
        print(c)

    # calculate and display output
    forward_prop = function([X, s_prev, h_prev], h)
    outp_matx = forward_prop(X_np, s_prev_np, h_prev_np)
    outp_str = mat2str(outp_matx)
    print(outp_str)

if __name__ == "__main__":
    test_layer_chars()
