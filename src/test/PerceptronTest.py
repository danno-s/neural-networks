import unittest
from src.perceptron import Perceptron, RandomPerceptron, AND, OR, NAND
from random import random

perceptron_half = Perceptron([0.5])
biased_perceptron = Perceptron([0.5], bias=-1)
complex_perceptron = Perceptron([1, 1, 8], bias=3)

def evaluate_sum_circuit(values):
    x1, x2 = values

    xx = NAND.process([x1, x2])

    s = NAND.process([
        NAND.process([
            x1,
            xx
        ]),
        NAND.process([
            x2,
            xx
        ])
    ])
    
    c = NAND.process([
        xx,
        xx
    ])

    return [s, c]

class PerceptronTest(unittest.TestCase):
    def test_single_input(self):
        self.assertEqual(perceptron_half.process([1]), 1, "Half perceptron miscalculated")
        self.assertEqual(biased_perceptron.process([1]), 0, "Biased perceptron miscalculated")

    def test_multiple_input(self):
        self.assertEqual(complex_perceptron.process([3, 6, 5]), 1, "Complex perceptron mixcalculated on the positive case")
        self.assertEqual(complex_perceptron.process([-1, 1, -1]), 0, "Complex perceptron miscalculated on the negative case")

    def test_edge_case(self):
        self.assertEqual(perceptron_half.process([0]), 0, "Half perceptron said 1 when calculated value was equal to 0")
        self.assertEqual(complex_perceptron.process([-1, 1, -3/8]), 0, "Complex perceptron miscalculated on the negative case")

    def test_and_perceptron(self):
        self.assertEqual(AND.process([0, 0]), 0, "AND failed on (0, 0)")
        self.assertEqual(AND.process([1, 0]), 0, "AND failed on (1, 0)")
        self.assertEqual(AND.process([0, 1]), 0, "AND failed on (0, 1)")
        self.assertEqual(AND.process([1, 1]), 1, "AND failed on (1, 1)")

    def test_or_perceptron(self):
        self.assertEqual(OR.process([0, 0]), 0, "OR failed on (0, 0)")
        self.assertEqual(OR.process([1, 0]), 1, "OR failed on (1, 0)")
        self.assertEqual(OR.process([0, 1]), 1, "OR failed on (0, 1)")
        self.assertEqual(OR.process([1, 1]), 1, "OR failed on (1, 1)")

    def test_nand_perceptron(self):
        self.assertEqual(NAND.process([0, 0]), 1, "NAND failed on (0, 0)")
        self.assertEqual(NAND.process([1, 0]), 1, "NAND failed on (1, 0)")
        self.assertEqual(NAND.process([0, 1]), 1, "NAND failed on (0, 1)")
        self.assertEqual(NAND.process([1, 1]), 0, "NAND failed on (1, 1)")

    def test_sum_circuit(self):
        self.assertEqual(evaluate_sum_circuit([0, 0]), [0, 0], "SUM failed on (0, 0)")
        self.assertEqual(evaluate_sum_circuit([1, 0]), [1, 0], "SUM failed on (1, 0)")
        self.assertEqual(evaluate_sum_circuit([0, 1]), [1, 0], "SUM failed on (0, 1)")
        self.assertEqual(evaluate_sum_circuit([1, 1]), [0, 1], "SUM failed on (1, 1)")

    def test_training(self):
        # First, we create a random perceptron that will learn if a point
        # is above a line in a 2D plane or not.
        r = RandomPerceptron(2)

        # Then, we establish what line the perceptron will be checking against
        m = random() * 50 - 50
        n = random() * 50 - 50
        
        # Then, we generate a set of random inputs within a square of side 50, centered in the origin
        inputs = [
            [random() * 50 - 50, random() * 50 - 50] for _ in range(100)
        ]

        expecteds = [int(coord[1] > coord[0] * m + n) for coord in inputs]

        # Finally, we train the percetron with the generated inputs
        r.train(inputs, expecteds)

        # Then we check to see if it actually learned, by generating some new inputs and checking the success rate
        test_inputs = [
            [random() * 50 - 50, random() * 50 - 50] for _ in range(100)
        ]

        test_expecteds = [int(coord[1] > coord[0] * m + n) for coord in test_inputs]

        correct = 0

        for i, e in zip(inputs, test_expecteds):
            if e == r.process(i):
                correct += 1

        # We expect a 90% success rate for such a simple problem
        assert correct / 100.0 >= 0.9

if __name__ == '__main__':
    unittest.main()