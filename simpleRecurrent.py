from blocks.bricks.recurrent import BaseRecurrent, recurrent
from theano import tensor
from blocks import initialization
from blocks.bricks import Identity
from blocks.bricks.recurrent import SimpleRecurrent
import theano
import numpy 

class FeedbackRNN(BaseRecurrent):
     def __init__(self, dim, **kwargs):
         super(FeedbackRNN, self).__init__(**kwargs)
         self.dim = dim
         self.first_recurrent_layer = SimpleRecurrent(dim=self.dim, activation=Identity(), name='first_recurrent_layer',weights_init=initialization.Identity())
         self.second_recurrent_layer = SimpleRecurrent(dim=self.dim, activation=Identity(), name='second_recurrent_layer',weights_init=initialization.Identity())
         self.children = [self.first_recurrent_layer,self.second_recurrent_layer]

     @recurrent(sequences=['inputs'], contexts=[],states=['first_states', 'second_states'],outputs=['first_states', 'second_states'])
     def apply(self, inputs, first_states=None, second_states=None):
         first_h = self.first_recurrent_layer.apply(inputs=inputs, states=first_states + second_states, iterate=False)
         second_h = self.second_recurrent_layer.apply(inputs=first_h, states=second_states, iterate=False)
         return first_h, second_h

     def get_dim(self, name):
         return (self.dim if name in ('inputs', 'first_states', 'second_states') else super(FeedbackRNN, self).get_dim(name))

x = tensor.tensor3('x')
feedback = FeedbackRNN(dim=3)
feedback.initialize()
first_h, second_h = feedback.apply(inputs=x)
f = theano.function([x], [first_h, second_h])
for states in f(numpy.ones((3, 1, 3), dtype=theano.config.floatX)):
    print(states) 