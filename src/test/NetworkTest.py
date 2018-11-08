import unittest

from src.network import Network
from src.layer import Layer
from random import random
from math import sin

singleNeuronNetwork = Network([
    Layer(5, 1)
])

multipleLayerNetwork = Network([
    Layer(10, 3),
    Layer(3, 5),
    Layer(5, 2),
    Layer(2, 10)
])

sineWaveNetwork = Network([
    Layer(2, 1)
])

class NetworkTest(unittest.TestCase):
    def testSingleNeuronNetwork(self):
        self.assertEqual(1, len(singleNeuronNetwork.first_layer))
        self.assertEqual(5, singleNeuronNetwork.first_layer.inputs)
        # Check if it can process correctly
        res = singleNeuronNetwork.process([1,2,3,4,5])
        self.assertEqual(1, len(res))

    def testMultipleLayerNetwork(self):
        self.assertEqual(3, len(multipleLayerNetwork.first_layer))
        self.assertEqual(10, multipleLayerNetwork.first_layer.inputs)
        # Check if it can process correctly
        res = multipleLayerNetwork.process([1,2,3,4,5,6,7,8,9,0])
        self.assertEqual(10, len(res))

    def testSineWave(self):
        # Generate a set of random inputs within a square of side 50, centered in the origin
        inputs = [[random() * 50 - 50, random() * 50 - 50] for _ in range(500)]

        expecteds = [[int(coord[1] > sin(coord[0]))] for coord in inputs]

        # Train the percetron with the generated inputs
        sineWaveNetwork.train(inputs, expecteds)

        # Then we check to see if it actually learned, by generating some new inputs and checking the success rate
        test_inputs = [[random() * 50 - 50, random() * 50 - 50] for _ in range(500)]

        test_expecteds = [[int(coord[1] > sin(coord[0]))] for coord in test_inputs]

        correct = 0

        for i, e in zip(inputs, test_expecteds):
            if e[0] == round(sineWaveNetwork.process(i)[0]):
                correct += 1

        # We expect a 95% success rate for such a simple problem
        self.assertGreaterEqual(correct / 500.0 , 0.95)


if __name__=='__main__':
    unittest.main()