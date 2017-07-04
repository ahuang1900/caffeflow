import sys

SHARED_CAFFE_RESOLVER = None

class CaffeResolver(object):
    def __init__(self):
        self.caffe = None
        self.use_caffe = False
        try:
            # Try to import PyCaffe first
            import caffe
            self.caffe = caffe
            self.caffepb = self.caffe.proto.caffe_pb2
            self.use_caffe = True
        except Exception as e:
            # Fall back to the protobuf implementation
            from . import caffe_pb2
            self.caffepb = caffe_pb2
            sys.stderr.write("INFO: There was an error when trying to import PyCaffe: {}\n"
                             "Falling back to the bundled protobuf.\n\n".format(e))
        self.NetParameter = self.caffepb.NetParameter

    def has_pycaffe(self):
        return self.use_caffe


def get_caffe_resolver():
    global SHARED_CAFFE_RESOLVER
    if SHARED_CAFFE_RESOLVER is None:
        SHARED_CAFFE_RESOLVER = CaffeResolver()
    return SHARED_CAFFE_RESOLVER


def has_pycaffe():
    return get_caffe_resolver().has_pycaffe()
