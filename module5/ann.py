# -*- encoding: utf-8 -*-

import logging

import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from module5.mnist.mnist_basics import load_all_flat_cases

# DEFAULTS
NETWORK_CONFIG = {
    'learning_rate': 0.001,
    'num_labels': 10,
    'max_training_runs': 60000,
    'max_test_runs': 10000,
    'training_batch_size': 512,
    'noise_dropout_input': 0.2,
    'noise_dropout_hidden': 0.5,
    'dropout_input': 0.0,
    'dropout_hidden': 0.0,
    'rho': 0.9,
    'epsilon': 1e-6
}

# SETTINGS
DATASET_DIR = 'module5'


def float_x(x):
    """
    Transform values in vector/list x to numPy array of floats
    :param x: A list / vector
    :return: A numpy array of type float
    """

    return np.asarray(x, dtype=theano.config.floatX)


def init_weights(shape):
    """
    Initialize a weight vector, given a shape of (x, y, z ...) dimensions, with each dimension representing node count.
    :param shape: A dimensional vector of (x, y, z ...). E.g (8, 8, 16, 8)
    :return: A weight mapping matrix from dim1 to dim2.
    """

    return theano.shared(float_x(np.random.randn(*shape) * 0.01))


def rectify(x):
    """
    Performs a vector adjustment of vector x, applying a lower bound of 0.0 for the value.
    :param x: An input value x
    :return: x if x > 0.0
    """

    return tensor.maximum(x, 0.)


def dropout(x, p=0., rng=None):
    """
    Adjusts the values of vector x, given a retain probability P
    :param x: A numpy vector/array
    :param p: A retain probability of 0 < p <= 1
    :param rng: A random number generator stream
    :return: A probability adjusted numpy vector
    """

    if not rng:
        rng = RandomStreams()
    if p > 0:
        retain_prob = 1 - p
        x *= rng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX)
        x /= retain_prob
    return x


def relu(x):
    """
    Performs the ReLU activation function on given value x
    :param x: The dot product between two vectors
    :return: ReLU value
    """

    return x * (x > 0)


def softmax(x):
    """
    Performs the softmax function on the value x
    :param x: The dot product between two vectors
    :return: Softmax value
    """

    e_x = tensor.exp(x - x.max(axis=1).dimshuffle(0, 'x'))

    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def generate_network_model(input_weight_vector, p_dropout_input, p_dropout_hidden, layers, activation_functions):
    """
    Generates a network model that links an input vector through layers, given dropout values for input/hidden.
    Activation functions for each layer are specified in the activation_functions list, and is index-correlated
    with the layers list.

    :param input_weight_vector: A numpy fmatrix of input the layer
    :param p_dropout_input: The dropout probability level for input nodes
    :param p_dropout_hidden: The dropout probability level for hidden layer nodes
    :param layers: A series of weight vectors for the hidden layers as well as the output layer
    :param activation_functions: A series of function pointers to activation functions
    :return: A tuple of hidden layers, followed by the output layer partial y with respect to x
    """

    if len(layers) != len(activation_functions) - 1:
        raise ValueError('The layers and activation function lists do not match in length!')

    output_weight_vector = layers.pop()
    output_activation_function = activation_functions.pop()
    output = list()

    input_layer = dropout(input_weight_vector, p_dropout_input)
    previous_layer = input_layer
    while layers:
        next_weight_matrix = layers.pop(0)
        next_activation_function = activation_functions.pop(0)
        layer = next_activation_function(tensor.dot(previous_layer, next_weight_matrix))
        previous_layer = dropout(layer, p_dropout_hidden)

        # Add layer to output after final dropout
        output.append(previous_layer)

    output_layer = output_activation_function(tensor.dot(previous_layer, output_weight_vector))
    output.append(output_layer)

    return np.array(output)


def rms_prop(_cost, _params, lr=NETWORK_CONFIG['learning_rate'],
             rho=NETWORK_CONFIG['rho'], epsilon=NETWORK_CONFIG['epsilon']):
    """
    Backpropagation error evaluation and correction
    :param _cost: Error correction function (We use crossentropy)
    :param _params: Weight matrix (E.g list of weight vectors for hidden and output layers)
    :param lr: Learning rate of the network
    :param rho: Greek letter Rho (Something to do with momentum)
    :param epsilon: Greek letter epsilon (Something to do with adjusting gradient scaling)
    :return: A vector of update values
    """

    _grads = tensor.grad(cost=_cost, wrt=_params)
    _updates = []

    for p, g in zip(_params, _grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = tensor.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        _updates.append((acc, acc_new))
        _updates.append((p, p - lr * g))

    return _updates


def default_backprop_updates(cost, params, lr=NETWORK_CONFIG['learning_rate']):
    """
    Default backpropagation update function
    :param cost: Cost function (cat crossentropy)
    :param params: Network model
    :return: List of updates
    """

    grads = tensor.grad(cost=cost, wrt=params)
    return [[p, p - g * lr] for p, g in zip(params, grads)]


def normalize_data(data, max_value=255):
    """
    Converts an input list/matrix to numpy array and adjusts
    :param data: The input data list/vector
    :return: A numpy array of floats adjusted based off max value
    """

    return np.array(data).astype('float64') / max_value


class ANN(object):
    """
    Construct an Artificial neural network
    """

    def __init__(self, layer_structure, activation_functions, config=NETWORK_CONFIG):
        """
        Constructs an Artificial Neural Network based on the provided argument properties.

        :param layer_structure: A list representing the number of (fully) connected nods in each layer.
        Example: [784, 620, 10], which has 784 input nodes, 620 nodes in a single hodden layer, and 10 output nodes.
        :param config: A Network Configuration dictionary, with a structure identical to the NETWORK_CONFIG defaults.
        """

        self._log = logging.getLogger(__name__)
        self._log.info('Initializing Neural Network...')
        self.train_input_data = []
        self.train_correct_labels = []
        self.test_input_data = []
        self.test_correct_labels = []
        self._network = None
        self._data_matrix = None
        self._label_matrix = None
        self._train = lambda x, y: False
        self._predict = lambda x: False

        # Update with new values if other than default is provided
        self._config = NETWORK_CONFIG
        self._config.update(config)

        self._log.info(
            'Layers: %s, Activation functions: %s, LR: %.5f, MTR: %d, MTE: %d, TRBS: %d.' % (
                repr(layer_structure),
                repr(activation_functions),
                self._config['learning_rate'],
                self._config['max_training_runs'],
                self._config['max_test_runs'],
                self._config['training_batch_size']
            )
        )

        self._srng = RandomStreams()
        self._generate_network(layer_structure, activation_functions)

    def _generate_network(self, layer_structure, activation_functions):
        """
        Generates the necessary function constructs needed to create a neural network.
        :param layer_structure: The list of node counts representing input, hidden and output layers.
        :param activation_functions: The list of activation functions corresponding to the layer structure
        """

        self._log.info('Generating network...')

        self._data_matrix = tensor.fmatrix()
        self._label_matrix = tensor.fmatrix()

        self._log.info('Generating weight matrices...')

        # Generate connection patterns between layers
        weight_matrix = list(
            init_weights((layer_structure[i], layer_structure[i + 1])) for i in range(len(layer_structure) - 1)
        )

        self._log.debug(weight_matrix)

        self._log.info('Generating noise model...')
        noise = generate_network_model(
            self._data_matrix,
            self._config['noise_dropout_input'],
            self._config['noise_dropout_hidden'],
            weight_matrix[:],
            activation_functions[:]
        )

        assert len(noise) == len(layer_structure) - 1

        self._log.info('Generating regular model...')
        layers = generate_network_model(
            self._data_matrix,
            self._config['dropout_input'],
            self._config['dropout_hidden'],
            weight_matrix[:],
            activation_functions[:]
        )

        assert len(layers) == len(layer_structure) - 1

        noise_output = noise[-1]
        regular_output = layers[-1]

        # Set up the output vector function
        output_function = tensor.argmax(regular_output, axis=1)
        # Set error correction function as crossentropy
        cost = tensor.mean(tensor.nnet.categorical_crossentropy(noise_output, self._label_matrix))
        params = weight_matrix
        # Updatefunction is backpropagaction algorithm using the specified error correction function
        updates = default_backprop_updates(cost, params, lr=self._config['learning_rate'])

        # Inject the training function that is used by self.train(*args)
        self._train = theano.function(
            inputs=[self._data_matrix, self._label_matrix],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True
        )
        # Inject the prediction function that is used by self.predict(*args)
        self._predict = theano.function(inputs=[self._data_matrix], outputs=output_function, allow_input_downcast=True)

    def train(self, epochs=0):
        """
        Train this network given a data
        :param epochs: The amount of iterations of training that should be done. Uses config training batch size.
        """

        self._log.info('Networking training initiated...')

        if not epochs:
            epochs = self._config['max_training_runs']

        for i in range(epochs):
            self._log.info('Training epoch: %d of %d' % (i, epochs))

            start_range = range(0, len(self.train_input_data), self._config['training_batch_size'])
            end_range = range(
                self._config['training_batch_size'],
                len(self.train_input_data),
                self._config['training_batch_size']
            )

            # Perform the actual training
            for start, end in zip(start_range, end_range):
                self._train(self.train_input_data[start:end], self.train_correct_labels[start:end])

            # Assess the correctness of the network on the entire test set after this epoch
            self._log.info(
                'Current epoch correctness: %.4f' % np.average(
                    np.argmax(self.test_correct_labels, axis=1) == self.predict(self.test_input_data)
                )
            )

    def predict(self, *args):
        """
        Attempt to predict a label vector based off the input data vector
        :param args: An input data vector
        :return: A labelvector the network has generated based off of the input vector
        """

        return self._predict(*args)

    def load_input_data(self, pickle_file=None):
        """
        Loads input data, either from default file or from specified gzipped pickle.
        NB: ALL input data is presumed to be as a flat structure.
        :param pickle_file: Path to a gzipped pickle file
        """

        self._log.info('Loading input data... (Existing file: %s)' % pickle_file)

        if pickle_file:
            pass
        else:
            self.train_input_data, self.train_correct_labels = load_all_flat_cases()
            self.test_input_data, self.test_correct_labels = load_all_flat_cases(type='testing')

        self._normalize_input_data()

    def _normalize_input_data(self):
        """
        Performs normalization in input data vectors, scaling them down to values in the range 0.0 - 1.0
        """

        self._log.debug(
            'Normalizing data and creating label vectors (Class count: %d)' % self._config['num_labels']
        )

        self.train_input_data = normalize_data(self.train_input_data)
        self.test_input_data = normalize_data(self.test_input_data)
        self.train_correct_labels = self.vectorize_labels(self.train_correct_labels, self._config['num_labels'])
        self.test_correct_labels = self.vectorize_labels(self.test_correct_labels, self._config['num_labels'])

    def blind_test(self, feature_sets):
        """
        The method blind test must accept a list of sublists, where each sublist is a vector of length 784 corresponding
        to the raw features of one image: each sublist is a flattened image containing integers in the range [0, 255].
        These raw features come directly from a flat-case file. The list does NOT contain labels, hence the adjective
        blind. Your method must produce a flat list of labels predicted by the ann when given each feature vector as
        input. Items in the labels list must correspond to items in the flattened image list. So if feature sets
        consists of five image vectors, a 7, two 3’s, an 8 and a 2 (in that order), then if your ann classifies them
        correctly, it should return this:

        [7,3,3,8,2]

        :param feature_sets: A list of sublists, with vectors of length 784 containing raw features of one image
        :return: A list containing the predicted values of each image
        """

        if not isinstance(feature_sets, list):
            raise ValueError('feature_sets must be an instance of list()')

        self._log.info('Initiating blind test with %d classification tasks...' % len(feature_sets))

        # Normalize the data to [0, 1] instead of [0, 255]
        feature_sets = normalize_data(feature_sets)

        raw_results = self.predict(feature_sets)

        """
        tolist is probably redundant here, but have done it to return
        on the format specified in the task description
        """
        return raw_results.tolist()

    @staticmethod
    def vectorize_labels(labels, num_categories):
        if type(labels) == list:
            labels = np.array(labels)
        labels = labels.flatten()
        label_vector = np.zeros((len(labels), num_categories))
        label_vector[np.arange(len(labels)), labels] = 1

        return label_vector
