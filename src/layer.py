from src.perceptron import SigmoidNeuron

class Layer():
    def __init__(self, inputs, nodes, next_layer=None):
        self.nodes = [SigmoidNeuron(inputs) for _ in range(nodes)]
        self.inputs = inputs
        self.next_layer = next_layer

    def process(self, inputs):
        assert len(inputs) == self.inputs, "Mismatched inputs in layer, should be {} not {}".format(self.inputs, len(inputs))
        
        results = []

        for node in self.nodes:
            results.append(node.process(inputs))

        if not self.next_layer:
            return results
        else:
            return self.next_layer.process(results)

    def __len__(self):
        return len(self.nodes)
