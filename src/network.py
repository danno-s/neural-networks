from itertools import tee
from src.layer import Layer

class Network():
    def __init__(self, layers):
        '''Receives a list of disconnected layers, and connects them in order. Only keeps a reference to the first layer
        '''
        self.first_layer = layers[0]

        def pairwise(iterable):
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        for layer, next_layer in pairwise(layers):
            layer.next_layer = next_layer

    def process(self, inputs):
        return self.first_layer.process(inputs)

