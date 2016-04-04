from theano import tensor
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks import MLP
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks_extras.extensions.plot import Plot  

x = tensor.matrix('features') # define the input variables
#initial parameters, define the input parameters linearly
input_to_hidden = Linear(name = 'input_to_hidden', input_dim = 784, output_dim = 100) # input_to_hidden is a matrix parameters, row = input_dim column = output_dim
h = Rectifier().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output', input_dim=100, output_dim=10)
#y_hat is a temporary value for hidden state
y_hat = Softmax().apply(hidden_to_output.apply(h))
#define the lost function
y = tensor.lmatrix('targets')
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
# regularizate the cost function
cg = ComputationGraph(cost)
W1,W2 = VariableFilter(roles = [WEIGHT])(cg.variables)
cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

mlp = MLP(activations = [Rectifier(), Softmax()], dims = [784,100,10]).apply(x) # dimsdims define the number of units in each layer
#除了第一层外，每一层都有一个激活函数 
#initial the weight
input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()
print W1.get_value()

mnist = MNIST(("train",))

data_stream = Flatten(DataStream.default_stream(mnist, iteration_scheme= SequentialScheme(mnist.num_examples, batch_size = 256)))

algorithm = GradientDescent(cost = cost, parameters = cg.parameters, step_rule = Scale(learning_rate = 0.1)) # define to use gradient dscent to compute

mnist_test = MNIST(("test", ))
data_stream_test = Flatten(DataStream.default_stream(mnist_test, iteration_scheme = SequentialScheme(mnist_test.num_examples, batch_size = 1024)))

monitor = DataStreamMonitoring(variables=[cost], data_stream = data_stream_test, prefix = "test")
#主函数执行区
main_loop = MainLoop(data_stream = data_stream, algorithm = algorithm, extensions = [monitor, FinishAfter(after_n_epochs = 1), TrainingDataMonitoring([cost, cg.parameters], after_batch=True),Plot('Plotting example', channels=[['cost'], ['a']],after_batch=True)]) #

main_loop.run()



























