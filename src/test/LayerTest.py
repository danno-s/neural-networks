import unittest
from src.layer import Layer

basicLayer = Layer(5, 10)

class LayerTest(unittest.TestCase):
    def testLayerInitialization(self):
        self.assertEqual(5, basicLayer.inputs)
        self.assertEqual(10, len(basicLayer))

if __name__ == "__main__":
    unittest.main()
