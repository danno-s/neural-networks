import unittest

from src.network import Network
from src.layer import Layer

singleNeuronNetwork = Network([
    Layer(5, 1)
])

multipleLayerNetwork = Network([
    Layer(10, 3),
    Layer(3, 5),
    Layer(5, 2),
    Layer(2, 10)
])

class NetworkTest(unittest.TestCase):
    def testSingleNeuronNetwork(self):
        self.assertEqual(1, len(singleNeuronNetwork.first_layer))
        self.assertEqual(5, singleNeuronNetwork.first_layer.inputs)
        # Check if it can process correctly
        singleNeuronNetwork.process([1,2,3,4,5])

    def testMultipleLayerNetwork(self):
        self.assertEqual(3, len(multipleLayerNetwork.first_layer))
        self.assertEqual(10, multipleLayerNetwork.first_layer.inputs)
        # Check if it can process correctly
        multipleLayerNetwork.process([1,2,3,4,5,6,7,8,9,0])
