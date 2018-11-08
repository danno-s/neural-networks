from src.network import Network
from src.layer import Layer

if __name__=='__main__':
    # First, build the network
    network = Network([
        Layer(64, 128),
        Layer(128, 64),
        Layer(64, 64),
        Layer(64, 16),
        Layer(16, 10)
    ])
    
    def generate_inputs(file):
        training = open(file)
        for line in training:
            yield list(map(lambda i: int(i), line.split(',')[:64]))

    def generate_outputs(file):
        training = open(file)
        for line in training:
            value = int(line.split(',')[64]) 
            yield [0 if i != value else 1 for i in range(10)]
    
    print("Training...", end="\r")

    # Train the network
    network.train(generate_inputs("optdigits.tra"), generate_outputs("optdigits.tra"))

    print("Testing...", end="\r")

    # Test the accuracy
    correct = 0
    total = 0
    for inputs, expected in zip(generate_inputs("optdigits.tes"), generate_outputs("optdigits.tes")):
        output = network.process(inputs)
        max_value = output.index(max(output))
        value = expected.index(1)
        if value == max_value:
            correct += 1
        total += 1

    print("La red tuvo {} de {} correctos. ({:.2%})".format(correct, total, correct / total))