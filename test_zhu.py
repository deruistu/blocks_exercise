# -*- coding: utf-8 -*-

import numpy as np
import logging
import numpy
import os
import time
import gzip


import argparse
import pprint

import configurations

from contextlib import closing
from six.moves import cPickle

from blocks.extensions.saveload import SAVED_TO, LOADED_FROM
from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.serialization import secure_dump, load, BRICK_DELIMITER
from blocks.utils import reraise_as
from blocks.filter import VariableFilter
from blocks.search import BeamSearch

from checkpoint import SaveLoadUtils
from sampling import SamplingBase

from stream import get_tr_stream, get_dev_stream
from theano import tensor,function

from model import BidirectionalEncoder, Decoder

from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.model import Model



logger = logging.getLogger(__name__)


class loadNMTfromFile(SaveLoadUtils):
    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        #super(loadNMTfromFile, self).__init__(saveto, **kwargs)

    def load_parameters(self):
        print("enter load_parameters")    
        #print("enter load_parameters")
        return self.load_parameter_values(self.path_to_parameters)

    def load_to(self, model):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        print "Loading model"
        try:
            logger.info(" ...loading model parameters")
            print "loading model parameters"
            params_all = self.load_parameters()
            params_this = model.get_parameter_dict()  
            missing = set(params_this.keys()) - set(params_all.keys())
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    if params_this[pname].get_value().shape != val.shape:
                        logger.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))
                        print "Dimension mismatch"

                    params_this[pname].set_value(val)
                    logger.info(" Loaded to CG {:15}: {}"
                                .format(val.shape, pname))
                    print "loaded to CG {:15}: {}".format(val.shape, pname)
                else:
                    logger.warning(" Parameter does not exist: {}".format(pname))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
        return model


class translateSentence(SamplingBase):
    def __init__(self,config, model,data_stream,hook_samples=1,src_vocab=None,trg_vocab=None,src_ivocab=None,trg_ivocab=None,src_vocab_size=None,**kwargs):
        #super(translateSentence,self).__init__(**kwargs)
        self.model = model
        self.hook_samples=1
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()
        self.config = config


    def getAlignment(self):
        unk_idx = self.config['unk_id']
        source_sentence = tensor.lmatrix('source')
        target_sentence = tensor.lmatrix('target')

        ftrans = open('/Users/lqy/Documents/transout.txt','w',0)

        falign = gzip.open('/Users/lqy/Documents/alignmentout','w',0)

        sampling_representation = encoder.apply(source_sentence, tensor.ones(source_sentence.shape))

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            seq = self._oov_to_unk(line[0], self.config['src_vocab_size'], unk_idx)
            input_ = numpy.tile(seq, (config['beam_size'], 1))
            print "input_: ",input_

    def initialValue(self,train_stream):

        sources = self._get_attr_rec(train_stream, 'data_stream')

        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

    def randomTranslate(self,train_stream,*args):
        print "enter into translate "
       
        sources = self._get_attr_rec(train_stream, 'data_stream')

        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        print "length:", self.src_vocab_size

        print "content: ",train_stream.mask_sources[0] 

        batch = args[0]
        #batch = train_stream.get_data_from_batch()   # what is the batch
        #batch = train_stream.data_stream.get_data(1)
        print "successfully"

        batch_size = batch['source'].shape[0]

        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            _1, outputs, _2, _3, costs = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            print("Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab))
            print("Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample cost: ", costs[:sample_length].sum())
            print()


def mergeSplit(sentence):
    new_result = []
    result = sentence.split()
    i = 0
    while i  <  len(result):
        if result[i][0] == '$':
            i = i + 2
            str = ''
            while result[i] != '}':
                str = str + ' ' + result[i]
                i += 1
            new_result += [str]
        else:
            new_result += [result[i]]
        i +=1
    return new_result

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config_cs2en",
                    help="Prototype config to use for config")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")
args = parser.parse_args()

#get configuration
config = getattr(configurations, args.proto)()

tr_stream = get_tr_stream(**config)
validate_stream = get_dev_stream(**config)

for i in range(1):

    logger.info('Creating theano variables')
    print("create theano variables")
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask') # what is the source_mask
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    #sampling_input = tensor.lmatrix('input')
# Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2)
    #cost = decoder.cost(encoder.apply(source_sentence, source_sentence_mask), # here source_sentence_mask 是embeding向量矩阵？ 属于自由向量？
    #        source_sentence_mask, target_sentence, target_sentence_mask)  # 定义cost 函数

    cost = decoder.cost(encoder.apply(source_sentence, tensor.ones(source_sentence.shape)),tensor.ones(source_sentence.shape), target_sentence, tensor.ones(target_sentence.shape))
    
    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)  # construct the computational graph for gradient computing. it aims to optimize the model，cg包含有整个完整运算的各个权值
    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
    config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()  # push_initialization_config 已经被预先定义在Initializable里的方法
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    sampling_representation = encoder.apply( source_sentence, tensor.ones(source_sentence.shape))
    generated = decoder.generate(source_sentence, sampling_representation)  # modified here to add the functions.
    search_model = Model(generated)

    _, samples = VariableFilter(bricks=[decoder.sequence_generator], name="outputs")(ComputationGraph(generated[1]))

    weights = VariableFilter(bricks=[decoder.sequence_generator],name="weights")(cg.variables)
    getAlignment = function([source_sentence, target_sentence], weights)
    beam_search = BeamSearch(samples=samples)

    
    saveTo = "/Users/lqy/Documents/search_model_fr2en_backup/"
    load_model = loadNMTfromFile(saveTo)
    model = load_model.load_to(search_model)
    nmt = translateSentence(config=config,model=search_model, data_stream=validate_stream,
                            hook_samples=config['hook_samples'],
                            every_n_batches=config['sampling_freq'],
                            src_vocab_size=config['src_vocab_size'])

    nmt.initialValue(tr_stream)

    unk_idx = config['unk_id']
    src_eos_idx = config['src_vocab_size'] - 1
    trg_eos_idx = config['trg_vocab_size'] - 1

    ftrans = open('/Users/lqy/Documents/transout.txt','w',0)

    falign = gzip.open('/Users/lqy/Documents/alignmentout','w',0)

        
    for i, line in enumerate(validate_stream.get_epoch_iterator()):
        source_line = line[0]
        #line_tok = mergeSplit(source_token[i])
        seq = nmt._oov_to_unk(line[0], config['src_vocab_size'], unk_idx)
        input_ = numpy.tile(seq, (config['beam_size'], 1)) #产生12 行1列的元素矩阵，元素指的是一个的序列
        #print "input_: ",input_[3]
        trans,costs = beam_search.search(input_values={source_sentence: input_[:]},max_length=3*len(seq), eol_symbol=src_eos_idx,ignore_first_eol=True)

        lengths = numpy.array([len(s) for s in trans])
        costs = costs / lengths

        best = numpy.argsort(costs)[0]

        trans_out = trans[best]

        source_word = nmt._idx_to_word(line[0],nmt.src_ivocab)
        trans_out_word = nmt._idx_to_word(trans_out, nmt.trg_ivocab)
        trans_out_word_str = trans_out_word.split(" ")
        source_word_str = source_word.split(" ")

        alignment = numpy.asarray(getAlignment(numpy.array(source_line)[None, :],numpy.array(trans_out)[None, :]))

        print "alignement shape is {}".format(alignment.shape)
        for i, word in enumerate(trans_out_word_str):
            
            align = numpy.argmax(alignment[0, int(i), 0, :])   # create a list, if there are more than two max
                #logger.info("alignment of the {} words in the target sentence'".format(align[0]))
            
            print "align is {}".format(align)
            print "probabilities is {}".format(alignment[0, int(i), 0, :])
            #print "the target word [{}] is translated by the source word {} with probabilities {}".format(word,source_word_str[align], alignment[0,int(i),0,align])



        print "source sentence:",source_word
        print "source sentence vector:",seq
        print "translate one:",trans_out_word
        print "translate vector:",trans_out
        print "cost:",costs[best] 
        #print "trans_out:",trans_out


    print "translation finished"
