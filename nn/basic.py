'''
    This file contains implementations of various NN components, including
      -- Dropout
      -- Feedforward layer (with customizable activations)
      -- RNN (with customizable activations)
      -- LSTM
      -- GRU
      -- CNN
    Each instance has a forward() method which takes x as input and return the
    post-activation representation y;
    Recurrent layers has two forward methods implemented:
        -- forward(x_t, h_tm1):  one step of forward given input x and previous
                                 hidden state h_tm1; return next hidden state
        -- forward_all(x, h_0):  apply successively steps given all inputs and
                                 initial hidden state, and return all hidden
                                 states h1, ..., h_n
'''

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from utils import say
from .initialization import default_srng, default_rng, USE_XAVIER_INIT
from .initialization import set_default_rng_seed, random_init, create_shared
from .initialization import ReLU, sigmoid, tanh, softmax, linear, softplus, get_activation_by_name

class Dropout(object):
    '''
        Dropout layer. forward(x) returns the dropout version of x
        Inputs
        ------
        dropout_prob : theano shared variable that stores the dropout probability
        srng         : theano random stream or None (default rng will be used)
        v2           : which dropout version to use
    '''
    def __init__(self, dropout_prob, srng=None, v2=False, return_mask=False):
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2
        self.return_mask = return_mask

    def forward(self, x):
        d = (1-self.dropout_prob) if not self.v2 else (1-self.dropout_prob)**0.5
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = x.shape,
                dtype = theano.config.floatX
            )
        if self.return_mask:
            return [x*mask/d, mask/d]
        else:
            return x*mask/d



def apply_dropout(x, dropout_prob, v2=False, return_mask=False):
    '''
        Apply dropout on x with the specified probability
    '''
    return Dropout(dropout_prob, v2=v2, return_mask=return_mask).forward(x)

def get_corrupted_input(x, corruption_level):
    """This function keeps ``1-corruption_level`` entries of the inputs the
    same and zero-out randomly selected subset of size ``coruption_level``
    Note : first argument of theano.rng.binomial is the shape(size) of
           random numbers that it should produce
           second argument is the number of trials
           third argument is the probability of success of any trial

            this will produce an array of 0s and 1s where 1 has a
            probability of 1 - ``corruption_level`` and 0 with
            ``corruption_level``

            The binomial function return int64 data type by
            default.  int64 multiplicated by the input
            type(floatX) always return float64.  To keep all data
            in floatX when floatX is float32, we set the dtype of
            the binomial to floatX. As in our case the value of
            the binomial is always 0 or 1, this don't change the
            result. This is needed to allow the gpu to work
            correctly as it only support float32 for now.

    """
    return default_srng.binomial(size=x.shape, n=1,
                                    p=1 - corruption_level,
                                    dtype=theano.config.floatX) * x

def apply_gaussian_noise(x, std_dev):
    noise = default_srng.normal(size=x.shape, std=std_dev, dtype=theano.config.floatX)
    return x + noise

class Layer(object):
    '''
        Basic neural layer -- y = f(Wx+b)
        foward(x) returns y
        Inputs
        ------
        n_in            : input dimension
        n_out           : output dimension
        activation      : the non-linear activation function to apply
        has_bias        : whether to include the bias term b in the computation
    '''
    def __init__(self, n_in, n_out, activation,
                            clip_gradients=False,
                            has_bias=True,
                            init_zero=False):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.has_bias = has_bias

        self.create_parameters(init_zero)

        # not implemented yet
        if clip_gradients is True:
            raise Exception("gradient clip not implemented")

    def create_parameters(self, init_zero=False):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.initialize_params(n_in, n_out, activation, init_zero)

    def initialize_params(self, n_in, n_out, activation, init_zero=False):
        if init_zero:
            b_vals = np.zeros((n_out,), dtype=theano.config.floatX)
            #b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            W_vals = np.zeros((n_in,n_out), dtype=theano.config.floatX)
        elif USE_XAVIER_INIT:
            if activation == ReLU:
                scale = np.sqrt(4.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            elif activation == softmax:
                scale = np.float64(0.001).astype(theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            else:
                scale = np.sqrt(2.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            W_vals = random_init((n_in,n_out), rng_type="normal") * scale
        else:
            W_vals = random_init((n_in,n_out))
            if activation == softmax:
                #W_vals *= 0.000
                W_vals = np.zeros((n_in,n_out), dtype=theano.config.floatX)
            if activation == ReLU:
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            else:
                b_vals = random_init((n_out,))
                #b_vals = np.zeros((n_out,), dtype=theano.config.floatX)
        self.W = create_shared(W_vals, name="W")
        if self.has_bias: self.b = create_shared(b_vals, name="b")

    def forward(self, x):
        if self.has_bias:
            return self.activation(
                    T.dot(x, self.W) + self.b
                )
        else:
            return self.activation(
                    T.dot(x, self.W)
                )
            
    def set_runmode(self, run_mode):
        pass
    
    def get_updates(self):
        return []

    @property
    def params(self):
        if self.has_bias:
            return [ self.W, self.b ]
        else:
            return [ self.W ]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())

class BNLayer(Layer):
    '''
        Batch normalization layer
        Inputs
        ------
        n_in            : input dimension
        n_out           : hidden dimension
        activation      : the non-linear function to apply
    '''
    def __init__(self, n_in, n_out, activation,
                            clip_gradients=False,
                            init_zero=False):
        super(BNLayer, self).__init__(
                n_in, n_out, activation,
                has_bias=False,
                clip_gradients = clip_gradients,
                init_zero=init_zero,
            )
        # (mini_batch_size, # features)
        self.BNLayer = BatchNormalization((None, n_out), mode=0)

    def forward(self, x):
        return self.activation(
                self.BNLayer.get_result(T.dot(x, self.W))
            )

    def set_runmode(self, run_mode) :
        self.BNLayer.set_runmode(run_mode) 

    def get_updates(self):
        return self.BNLayer.updates

    @property
    def params(self):
        return [ self.W ] + self.BNLayer.params

    @params.setter
    def params(self, param_list):
        raise Exception()
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())

class RecurrentLayer(Layer):
    '''
        Basic recurrent layer -- h_t = f(Wx + W'h_{t-1} + b)
            forward(x, h_{t-1}) executes one step of the RNN and returns h_t
            forward_all(x, h_0) executes all steps and returns H = {h_0, ... , h_n}
        Inputs
        ------
        n_in            : input dimension
        n_out           : hidden dimension
        activation      : the non-linear function to apply
    '''
    def __init__(self, n_in, n_out, activation,
            clip_gradients=False,
            init_zero=False):
        super(RecurrentLayer, self).__init__(
                n_in, n_out, activation,
                clip_gradients = clip_gradients,
                init_zero=init_zero,
            )

    def create_parameters(self, init_zero=False):
        print 'init_zero', init_zero
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        # re-use the code in super-class Layer
        self.initialize_params(n_in + n_out, n_out, activation, init_zero)

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        return activation(
                T.dot(x, self.W[:n_in]) + T.dot(h, self.W[n_in:]) + self.b
            )

    def forward_all(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        return h


class EmbeddingLayer(object):
    '''
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)
        Inputs
        ------
        n_d             : dimension of word embeddings; may be over-written if embs
                            is specified
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
        fix_init_embs   : whether to fix the initial word vectors loaded from embs
    '''
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):
        self.fix_init_embs = fix_init_embs
        print 'fix_init_embs:', fix_init_embs
        if embs is not None:
            vocab_map = {}
            emb_vals = [ ]
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            say("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
                    #emb_vals.append(random_init((n_d,)))

            emb_vals = np.vstack(emb_vals).astype(theano.config.floatX)
            recon_emb_vals = np.tanh(np.vstack(emb_vals)).astype(theano.config.floatX)
            self.vocab_map = vocab_map
        else:
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)

            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d))
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        self.embeddings = create_shared(emb_vals)
        self.recon_embeddings = create_shared(recon_emb_vals)
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        self.n_V = len(self.vocab_map)
        self.n_d = n_d
        say("Vocabulary size: {}.\n".format(self.n_V))
        
        self.id_to_word = [None] * self.n_V
        for w in self.vocab_map:
            self.id_to_word[self.vocab_map[w]] = w

    def map_to_ids(self, words, filter_oov=False):
        '''
            map the list of string tokens into a numpy array of integer IDs
            Inputs
            ------
            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array
            Outputs
            -------
            return the numpy array of word IDs
        '''
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )
            
    def get_words(self, ids):
        id_to_word = self.id_to_word
        return [id_to_word[i] for i in ids if id_to_word[i] != "<padding>"]

    def forward(self, x):
        '''
            Fetch and return the word embeddings given word IDs x
            Inputs
            ------
            x           : a theano array of integer IDs
            Outputs
            -------
            a theano matrix of word embeddings
        '''
        return self.embeddings[x]
    
    def recon_forward(self, x):
        return self.recon_embeddings[x]

    @property
    def params(self):
        return [ self.embeddings_trainable ]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())


class LSTM(Layer):
    '''
        LSTM implementation.
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False, init_zero=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients

        #self.in_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients, init_zero)
        #self.forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients, init_zero)
        #self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients, init_zero)
        self.in_gate = RecurrentLayer(n_in+n_out, n_out, sigmoid, clip_gradients, init_zero)
        self.out_gate = RecurrentLayer(n_in+n_out, n_out, sigmoid, clip_gradients, init_zero)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients, init_zero)


        self.internal_layers = [ self.input_layer, self.in_gate,
                                 self.out_gate]#,  self.forget_gate]

    def forward(self, x, hc):
        '''
            Apply one recurrent step of LSTM
            Inputs
            ------
                x       : the input vector or matrix
                hc      : the vector/matrix of [ c_tm1, h_tm1 ], i.e. hidden state and
                            visible state concatenated together
            Outputs
            -------
                return [ c_t, h_t ] as a single concatenated vector/matrix
        '''
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[:, :n_out]
            h_tm1 = hc[:, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        if hc.ndim > 1:
            xc_tm1 = T.concatenate([ x, c_tm1 ], axis=1)
        else:
            xc_tm1 = T.concatenate([ x, c_tm1 ])
        in_t = self.in_gate.forward(xc_tm1, h_tm1)
        c_t = (1 - in_t) * c_tm1 + in_t * self.input_layer.forward(x,h_tm1)
        if hc.ndim > 1:
            xc_t = T.concatenate([ x, c_t ], axis=1)
        else:
            xc_t = T.concatenate([ x, c_t ])
        out_t = self.out_gate.forward(xc_t, h_tm1)
        h_t = out_t * T.tanh(c_t)

        if hc.ndim > 1:
            return T.concatenate([ c_t, h_t ], axis=1)
        else:
            return T.concatenate([ c_t, h_t ])

    def forward_all(self, x, h0=None, return_c=False):
        '''
            Apply recurrent steps of LSTM on all inputs {x_1, ..., x_n}
            Inputs
            ------
            x           : input as a matrix (n*d) or a tensor (n*batch*d)
            h0          : the initial states [ c_0, h_0 ] including both hidden and
                            visible states
            return_c    : whether to return hidden state {c1, ..., c_n}
            Outputs
            -------
            if return_c is False, return {h_1, ..., h_n}, otherwise return
                { [c_1,h_1], ... , [c_n,h_n] }. Both represented as a matrix or tensor.
        '''
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*2,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out:]
        else:
            return h[:,self.n_out:]

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

class GRU(Layer):
    '''
        GRU implementation
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients

        self.reset_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.update_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)

        self.internal_layers = [ self.reset_gate, self.update_gate, self.input_layer ]

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        reset_t = self.reset_gate.forward(x, h)
        update_t = self.update_gate.forward(x, h)
        h_reset = reset_t * h

        h_new = self.input_layer.forward(x, h_reset)
        h_out = update_t*h_new + (1.0-update_t)*h
        return h_out

    def forward_all(self, x, h0=None, return_c=True):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        return h

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end


class LeCNN(Layer):
    '''
        CNN implementation. Return feature maps over time. No pooling is used.

        Inputs
        ------

            order       : feature filter width
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            order=1, clip_gradients=False, BN=0):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients

        # batch, in, row, col
        self.input_shape = (None, n_in, 1, None)
        # out, in, row, col
        self.filter_shape = (n_out, n_in, 1, order)
        self.W = create_shared(random_init(self.filter_shape), name="W")
        if BN == 0:
            self.bias = create_shared(random_init((n_out,)), name="bias")
        
        self.BNLayer = None
        self.BN = BN
        if BN > 0:
            # calculate appropriate input_shape, (mini_batch_size, # of channel, # row, # column)
            new_shape = list(self.input_shape)
            new_shape[1] = self.filter_shape[0]
            new_shape = tuple(new_shape)
            self.BNLayers = [BatchNormalization(new_shape, mode=1) for _ in xrange(BN)]

    def forward_all(self, x, domain=-1, create_updates=True, pad=None):

        # x is len*batch*d, xs is batch*d*1*len
        xs = x.dimshuffle((1,2,'x',0))

        # batch*d*1*len
        conv_out = T.nnet.conv2d(
                input = xs,
                filters = self.W,
                #image_shape = self.input_shape,
                input_shape = self.input_shape,
                filter_shape = self.filter_shape,
                border_mode = 'half'
        )
        if self.BN > 0:
            assert domain >= 0
            conv_out = self.BNLayers[domain].get_result(conv_out, create_updates)
        else:
            conv_out = conv_out + self.bias.dimshuffle(('x',0,'x','x'))
        conv_out = self.activation(conv_out)

        # batch*d*len
        h = conv_out.flatten(3)

        # len*batch*d
        return h.dimshuffle((2,0,1))

    def set_runmode(self, run_mode) :
        if self.BN > 0:
            for BNLayer in self.BNLayers:
                BNLayer.set_runmode(run_mode)
            
    def get_updates(self):
        if self.BN > 0:
            updates = []
            for BNLayer in self.BNLayers:
                updates += BNLayer.updates
            return updates
        else:
            return []

    @property
    def params(self):
        if self.BN > 0:
            params = [ self.W ]
            for BNLayer in self.BNLayers:
                params += BNLayer.params
        else:
            params = [ self.W, self.bias ]
        return params 

    @params.setter
    def params(self, param_list):
        raise Exception()
        self.W.set_value(param_list[0].get_value())
        self.bias.set_value(param_list[1].get_value())

class BatchNormalization(object) :
    def __init__(self, input_shape, mode=0 , momentum=0.9) :
        '''
        # params :
        input_shape :
            when mode is 0, we assume 2D input. (mini_batch_size, # features)
            when mode is 1, we assume 4D input. (mini_batch_size, # of channel, # row, # column)
        mode : 
            0 : feature-wise mode (normal BN)
            1 : window-wise mode (CNN mode BN)
        momentum : momentum for exponential average
        '''
        self.input_shape = input_shape
        self.mode = mode
        self.momentum = momentum
        #self.run_mode = 0 # run_mode : 0 means training, 1 means inference
        self.run_mode = theano.shared(np.float64(0.0).astype(theano.config.floatX))

        self.insize = input_shape[1]
        
        # random setting of gamma and beta, setting initial mean and std
        rng = default_rng
        self.gamma = create_shared(rng.uniform(low=-(1.0/self.insize)**0.5, high=(1.0/self.insize)**0.5, size=(input_shape[1])).astype(theano.config.floatX), name='gamma')
        self.beta = create_shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='beta')
        self.mean = create_shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='mean')
        self.var = create_shared(np.ones((input_shape[1]), dtype=theano.config.floatX), name='var')

        # parameter save for update
        self.params = [self.gamma, self.beta]
        self.updates = None

    def set_runmode(self, run_mode) :
        self.run_mode.set_value(np.float64(run_mode).astype(theano.config.floatX))

    def create_updates(self, input):
        if self.mode == 0:
            now_mean = T.mean(input, axis=0)
            now_var = T.var(input, axis=0)
            batch = T.cast(input.shape[0], theano.config.floatX)
        else:
            now_mean = T.mean(input, axis=(0,2,3))
            now_var = T.var(input, axis=(0,2,3))
            batch = T.cast(input.shape[0]*input.shape[2]*input.shape[3], theano.config.floatX)
        if self.updates is None:
            new_mean = self.momentum * self.mean + (1.0-self.momentum) * now_mean
            new_var = self.momentum * self.var + (1.0-self.momentum) * ((batch+1.0)/batch*now_var)
        else:
            new_mean = self.momentum * self.updates[0][1] + (1.0-self.momentum) * now_mean
            new_var = self.momentum * self.updates[1][1] + (1.0-self.momentum) * ((batch+1.0)/batch*now_var)
        self.updates = [(self.mean, new_mean), (self.var, new_var)]

    def get_result(self, input, create_updates) :
        if create_updates:
            self.create_updates(input)
        
        # returns BN result for given input.
        epsilon = np.float64(1e-06).astype(theano.config.floatX)

        if self.mode == 0:
            now_mean = T.mean(input, axis=0)
            now_var = T.var(input, axis=0)
        else:
            now_mean = T.mean(input, axis=(0,2,3))
            now_var = T.var(input, axis=(0,2,3))
        now_mean = self.run_mode * self.mean + (1.0-self.run_mode) * now_mean
        now_var = self.run_mode * self.var + (1.0-self.run_mode) * now_var
            
        if self.mode == 0:
            output = self.gamma * (input - now_mean) / (T.sqrt(now_var+epsilon)) + self.beta
        else:
            output = self.gamma.dimshuffle(('x', 0, 'x', 'x')) * (input - now_mean.dimshuffle(('x', 0, 'x', 'x'))) \
                    / (T.sqrt(now_var+epsilon).dimshuffle(('x', 0, 'x', 'x'))) + self.beta.dimshuffle(('x', 0, 'x', 'x'))
                        
        return output
