import unittest

from src.neuron import SigmoidNeuron


class NeuronTest(unittest.TestCase):
    def setUp(self):
        self.neuron = SigmoidNeuron(3, learning_rate = 0.5)
        # Set weight to a constant to be able to test
        self.neuron.weights = [1, 2, 3]
        self.neuron.bias = -5

    def testProcess(self):
        self.neuron.process([3,2,1])
        self.assertIsNotNone(self.neuron.output)

    def testDelta(self):
        self.neuron.process([3,2,1])
        self.neuron.adjust_delta(2)
        transfer_derivative = self.neuron.output * (1 - self.neuron.output)
        self.assertIsNotNone(self.neuron.delta)
        self.assertEqual(2 * transfer_derivative, self.neuron.delta)
    
    def testAdjustBias(self):
        self.neuron.process([3,2,1])
        self.neuron.adjust_delta(2)
        delta = self.neuron.delta
        self.neuron.adjust_bias()
        self.assertEqual(-5 + 0.5 * delta, self.neuron.bias)

    def testAdjustWeights(self):
        self.neuron.process([3,2,1])
        self.neuron.adjust_delta(2)
        delta = self.neuron.delta
        self.neuron.adjust_weights([3,2,1])
        self.assertEqual([1 + 0.5 * delta * 3, 2 + 0.5 * delta * 2, 3 + 0.5 * delta * 1], self.neuron.weights)
