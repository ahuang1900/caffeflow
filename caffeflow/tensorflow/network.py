import numpy as np
import tensorflow as tf


def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


def format_padding(pad_h, pad_w):
    return [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]


class Network(object):
    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        """Construct the network. """
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        """Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        """Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        """Returns the current network output."""
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return tf.get_variable(name, shape, trainable=self.trainable)

    @layer
    def conv(self,
             input_,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             pad_h=0,
             pad_w=0,
             group=1,
             biased=True):
        # Get the number of channels in the input
        c_i = input_.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0

        # Convolution for a given input and kernel
        def convolve(i_, k_):
            return tf.nn.conv2d(i_, k_, [1, s_h, s_w, 1], padding='VALID')

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                if pad_h or pad_w:
                    input_ = tf.pad(input_, format_padding(pad_h, pad_w))

                output = convolve(input_, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(input_, group, 3)

                if pad_h or pad_w:
                    input_groups = [tf.pad(i, format_padding(pad_h, pad_w)) for i in input_groups]

                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(output_groups, 3)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input_, name):
        return tf.nn.relu(input_, name=name)

    @layer
    def max_pool(self, input_, k_h, k_w, s_h, s_w, name, pad_h=0, pad_w=0):
        if pad_h or pad_w:
            input_ = tf.pad(input_, format_padding(pad_h, pad_w))

        return tf.nn.max_pool(input_,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding='VALID',
                              name=name)

    @layer
    def avg_pool(self, input_, k_h, k_w, s_h, s_w, name, pad_h=0, pad_w=0):
        if pad_h or pad_w:
            input_ = tf.pad(input_, format_padding(pad_h, pad_w))

        return tf.nn.avg_pool(input_,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding='VALID',
                              name=name)

    @layer
    def lrn(self, input_, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input_,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input_, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input_.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input_, [-1, dim])
            else:
                feed_in, dim = (input_, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input_, name):
        input_shape = map(lambda v: v.value, input_.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input_ = tf.squeeze(input_, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input_, name=name)

    @layer
    def batch_normalization(self, input_, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name):
            shape = [input_.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input_,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, input_, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input_, keep, name=name)
