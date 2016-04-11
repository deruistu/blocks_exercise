# 这个程序只有两层
from theano import tensor
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.algorithms import GradientDescent, Scale
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing




x = tensor.matrix("features")
input_to_hidden = Linear(name = "input_to_hidden", input_dim = 784, output_dim = 120) # define a function. 这个函数用于计算输入层到隐藏层的线性计算
h = Rectifier().apply(input_to_hidden.apply(x)) # 以每个隐藏层单元获得的线性计算结果为输入，计算使用Rectifier()激活函数的每一个隐藏层单元相应的输出结果，这个结果将被用于作为下一层的输入
hidden_to_output = Linear(name = "hidden_to_output", input_dim = 120, output_dim = 10) # 定义最终输出层的每一个单元的线性计算结果。
y_hat = Softmax().apply(hidden_to_output(h)) # 得出输出层每一个神经单元的非线性输出转换。

y = tensor.lmatrix("targer") # 定义输出变量

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat) #定义cost 函数

#构造计算图。
cg = ComputationGraph(cost)

#对cost函数进行正则化
#选择需要计算的参数 W1 为第一层所有线性转换的 W ，W2 为第二层所有线性转换的W
W1,W2 = VariableFilter(roles = [WEIGHT])(cg.variables)
#正则化公式定义，此处使用的是L2正则化
cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

#定义一个多层神经网络，层与层之间的计算公式已经被之前定义。
#激活函数集activations定义了每一层的非线性转换函数，多层感知器每一层的输出都包含了两部分，第一部分是线性计算，然后将线性计算的结果进行非线性转换
#x是多层感知器的输入
mlp = MLP(activations = [Rectifier(),Softmax()], dims = [784, 100, 10]).apply(x)

#定义完整个神经网络的流程后，需要设置其线性转换的参数的初始值。
input_to_hidden.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = Constant(0);
hidden_to_output.weights_init = IsotropicGaussian(0.01)
hidden_to_output.biases_init = Constant(0)

#对设置进行初始化设置，必须要做这一步，否则之前的设置都没有用
input_to_hidden.initialize()
hidden_to_output.initialize()

#然后开始进行模型训练，这里使用现有的内置的数据集 MNIST，如果想要使用别的数据集，需要使用fuel对数据进行预处理
mnist = MNIST(("train",))
#定义迭代计算的方式，使用mini-batch的方法计算，每一次mini-batch使用1024条数据。以此获得数据流，data_stream
data_stream = Flatten(DataStream.default_stream(mnist, iteration_scheme = SequentialScheme(mnist.num_example, batch_size = 256)))

#定义优化函数的最优值计算方法，这边使用SGD来做
algorithm = GradientDescent(cost = cost, parameter = [cg.parameters], step_rule = Scale(learning_rate = 0.01))


#对训练数据的指定参数进行监控,使用DataStreamMonitoring方法,使用test 集合来验证结果。在训练过程中，可以查看算法在test集合的性能表现。
mnist_test = MNIST(("test",))
data_stream_test = Flatten(DataStrea.default_stream(mnist_test, iteration_scheme = SequentialScheme(mnist_test.num_example, batch_size= 1024)))
monitor = DataStreamMonitoring(variables = [cost], data_stream = data_stream_test, prefix = "test")

#设置mainloop 循环进行计算 , MainLoop是blocks的核心功能
main_loop = MainLoop(data_stream = data_stream, algorithm = algorithm, extensions =[monitor, FinishAfter(after_n_epoch=1), Printing(after_batch = true)])

main_loop.run()








