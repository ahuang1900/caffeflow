from __future__ import absolute_import, division, print_function

import re
import numbers

from .shapes import *

LAYER_DESCRIPTORS = {

    # Caffe Types
    u'AbsVal': shape_identity,
    u'Accuracy': shape_scalar,
    u'ArgMax': shape_not_implemented,
    u'BatchNorm': shape_identity,
    u'BNLL': shape_not_implemented,
    u'Concat': shape_concat,
    u'ContrastiveLoss': shape_scalar,
    u'Convolution': shape_convolution,
    u'Deconvolution': shape_not_implemented,
    u'Data': shape_data,
    u'Dropout': shape_identity,
    u'DummyData': shape_data,
    u'EuclideanLoss': shape_scalar,
    u'Eltwise': shape_identity,
    u'Exp': shape_identity,
    u'Flatten': shape_not_implemented,
    u'HDF5Data': shape_data,
    u'HDF5Output': shape_identity,
    u'HingeLoss': shape_scalar,
    u'Im2col': shape_not_implemented,
    u'ImageData': shape_data,
    u'InfogainLoss': shape_scalar,
    u'InnerProduct': shape_inner_product,
    u'Input': shape_data,
    u'LRN': shape_identity,
    u'MemoryData': shape_mem_data,
    u'MultinomialLogisticLoss': shape_scalar,
    u'MVN': shape_not_implemented,
    u'Pooling': shape_pool,
    u'Power': shape_identity,
    u'ReLU': shape_identity,
    u'Scale': shape_identity,
    u'Sigmoid': shape_identity,
    u'SigmoidCrossEntropyLoss': shape_scalar,
    u'Silence': shape_not_implemented,
    u'Softmax': shape_identity,
    u'SoftmaxWithLoss': shape_scalar,
    u'Split': shape_not_implemented,
    u'Slice': shape_not_implemented,
    u'TanH': shape_identity,
    u'WindowData': shape_not_implemented,
    u'Threshold': shape_identity,
}

LAYER_TYPES = LAYER_DESCRIPTORS.keys()

# This string literal should be left in bytes in Python 2 and in unicode in Python 3; otherwise it won't work.
LayerType = type('LayerType', (), {t: t for t in LAYER_TYPES})


class NodeKind(LayerType):
    @staticmethod
    def map_raw_kind(kind):
        if kind in LAYER_TYPES:
            return kind
        return None

    @staticmethod
    def compute_output_shape(node):
        try:
            val = LAYER_DESCRIPTORS[node.kind](node)
            return val
        except NotImplementedError:
            raise KaffeError(u'Output shape computation not implemented for type: %s' % node.kind)


class NodeDispatchError(KaffeError):
    pass


class NodeDispatch(object):
    @staticmethod
    def get_handler_name(node_kind):
        if len(node_kind) <= 4:
            # A catch-all for things like ReLU and tanh
            return node_kind.lower()
        # Convert from CamelCase to under_scored
        name = re.sub(u'(.)([A-Z][a-z]+)', r'\1_\2', node_kind)
        return re.sub(u'([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def get_handler(self, node_kind, prefix):
        name = self.get_handler_name(node_kind)
        name = u'_'.join((prefix, name))
        try:
            return getattr(self, name)
        except AttributeError:
            raise NodeDispatchError(u'No handler found for node kind: %s (expected: %s)' %
                                    (node_kind, name))


class LayerAdapter(object):
    def __init__(self, layer, kind):
        self.layer = layer
        self.kind = kind

    @property
    def parameters(self):
        name = NodeDispatch.get_handler_name(self.kind)
        name = u'_'.join((name, u'param'))
        try:
            return getattr(self.layer, name)
        except AttributeError:
            raise NodeDispatchError(u'Caffe parameters not found for layer kind: {}'.format(self.kind))

    @staticmethod
    def get_kernel_value(scalar, repeated, idx, default=None):
        if scalar:
            return scalar
        if repeated:
            if isinstance(repeated, numbers.Number):
                return repeated
            if len(repeated) == 1:
                # Same value applies to all spatial dimensions
                return int(repeated[0])
            assert idx < len(repeated)
            # Extract the value for the given spatial dimension
            return repeated[idx]
        if default is None:
            raise ValueError(u'Unable to determine kernel parameter!')
        return default

    @property
    def kernel_parameters(self):
        assert self.kind in (NodeKind.Convolution, NodeKind.Pooling)
        params = self.parameters
        k_h = self.get_kernel_value(params.kernel_h, params.kernel_size, 0)
        k_w = self.get_kernel_value(params.kernel_w, params.kernel_size, 1)
        s_h = self.get_kernel_value(params.stride_h, params.stride, 0, default=1)
        s_w = self.get_kernel_value(params.stride_w, params.stride, 1, default=1)
        pad_h = self.get_kernel_value(params.pad_h, params.pad, 0, default=0)
        pad_w = self.get_kernel_value(params.pad_h, params.pad, 1, default=0)
        return KernelParameters(k_h, k_w, s_h, s_w, pad_h, pad_w)


KernelParameters = namedtuple(u'KernelParameters', [u'kernel_h', u'kernel_w', u'stride_h', u'stride_w',
                                                    u'pad_h', u'pad_w'])
