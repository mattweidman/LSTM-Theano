import numpy as np
import theano
import theano.tensor as T

class LSTM_layer:

    # n_in: size of input to this layer
    # n_out: size of output to this layer
    def __init__(self, n_in, n_out):

        # record input and output size
        self.input_size = n_in
        self.output_size = n_out

        # weights for input vector
        W_x = lambda s: theano.shared(np.random.randn(n_in, n_out), name=s)
        self.Wgx = W_x('Wgx')
        self.Wix = W_x('Wix')
        self.Wfx = W_x('Wfx')
        self.Wox = W_x('Wox')

        # weights for previous output vector
        W_h = lambda s: theano.shared(np.random.randn(n_out, n_out), name=s)
        self.Wgh = W_h('Wgh')
        self.Wih = W_h('Wih')
        self.Wfh = W_h('Wfh')
        self.Woh = W_h('Woh')

        # biases for each gate
        b_ = lambda s: theano.shared(np.random.randn(n_out), name=s)
        self.bg = b_('bg')
        self.bi = b_('bi')
        self.bf = b_('bf')
        self.bo = b_('bo')

    # forward propagate through this LSTM layer
    # X: input, size (num_examples, input_size), type theano.tensor.dmatrix
    # s_prev: previous internal state, size (num_examples, output_size), type
    # theano.tensor.dmatrix
    # h_prev: previous output, size (num_examples, output_size), type
    # theano.tensor.dmatrix
    def output(self, X, s_prev, h_prev):
        g = T.tanh(T.dot(X, self.Wgx)+T.dot(h_prev, self.Wgh)+self.bg)
        i = T.nnet.sigmoid(T.dot(X, self.Wix)+T.dot(h_prev, self.Wih)+self.bi)
        f = T.nnet.sigmoid(T.dot(X, self.Wfx)+T.dot(h_prev, self.Wfh)+self.bf)
        o = T.nnet.sigmoid(T.dot(X, self.Wox)+T.dot(h_prev, self.Woh)+self.bo)
        s = g*i + s_prev*f
        h = T.tanh(s) * o
        return s, h

    def get_param_list(self):
        return [self.Wgx, self.Wix, self.Wfx, self.Wox, self.Wgh, self.Wih,
            self.Wfh, self.Woh, self.bg, self.bi, self.bf, self.bo]
