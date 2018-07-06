import ROOT
from larcv import larcv
larcv.ThreadProcessor
from larcv.dataloader2 import larcv_threadio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys,time

# tensorflow/gpu start-up configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#env CUDA_DEVICE_ORDER=PCI_BUS_ID
#env CUDA_VISIBLE_DEVICES=2
import tensorflow as tf

sys.path.insert(0, 'u-resnet/lib')
import ssnet_trainval as api
t = api.ssnet_trainval()

io_config = \
"""
MainIO: {
  Verbosity:    3
  EnableFilter: false
  RandomAccess: 2
  RandomSeed:   123
  InputFiles:   ["data/practice_train_2k.root"]
  ProcessType:  ["BatchFillerImage2D","BatchFillerImage2D"]
  ProcessName:  ["main_data","main_label"]
  NumThreads: 1
  NumBatchStorage: 4

  ProcessList: {
    main_data: {
      Verbosity: 3
      ImageProducer: "data"
      Channels: [0]
    }
    main_label: {
      Verbosity: 3
      ImageProducer: "segment"
      Channels: [0]
    }
  }
}      
"""

import tempfile
train_io_config = tempfile.NamedTemporaryFile('w')
train_io_config.write(io_config)
train_io_config.flush()

io_config = \
"""
TestIO: {
  Verbosity:    3
  EnableFilter: false
  RandomAccess: 2
  RandomSeed:   123
  InputFiles:   ["data/practice_test_2k.root"]
  ProcessType:  ["BatchFillerImage2D","BatchFillerImage2D"]
  ProcessName:  ["test_data","test_label"]
  NumThreads: 1
  NumBatchStorage: 2

  ProcessList: {
    test_data: {
      Verbosity: 3
      ImageProducer: "data"
      Channels: [0]
    }
    test_label: {
      Verbosity: 3
      ImageProducer: "segment"
      Channels: [0]
    }
  }
}

"""

import tempfile
test_io_config = tempfile.NamedTemporaryFile('w')
test_io_config.write(io_config)
test_io_config.flush()



train_config = \
"""
NUM_CLASS          3
BASE_NUM_FILTERS   16
MAIN_INPUT_CONFIG  '{:s}'
TEST_INPUT_CONFIG  '{:s}'
LOGDIR             'ssnet_train_log'
SAVE_FILE          'ssnet_checkpoint/uresnet'
LOAD_FILE          ''
AVOID_LOAD_PARAMS  []
ITERATIONS         8000
MINIBATCH_SIZE     20
NUM_MINIBATCHES    1
DEBUG              False
TRAIN              True
TF_RANDOM_SEED     123
USE_WEIGHTS        False
REPORT_STEPS       200
SUMMARY_STEPS      20
CHECKPOINT_STEPS   100
CHECKPOINT_NMAX    20
CHECKPOINT_NHOUR   0.4
KEYWORD_DATA       'main_data'
KEYWORD_LABEL      'main_label'
KEYWORD_WEIGHT     ''
KEYWORD_TEST_DATA  'test_data'
KEYWORD_TEST_LABEL 'test_label'
KEYWORD_TEST_WEIGHT ''
"""

import tempfile
ssnet_config = tempfile.NamedTemporaryFile('w')
ssnet_config.write(train_config.format(train_io_config.name, test_io_config.name))
ssnet_config.flush()


t.override_config(ssnet_config.name)
t.initialize()

ENTRY=2

def get_entry(entry):
    # image
    chain_image2d = ROOT.TChain("image2d_data_tree")
    chain_image2d.AddFile('data/test_10k.root')
    chain_image2d.GetEntry(entry)
    cpp_image2d = chain_image2d.image2d_data_branch.as_vector().front()
    # label
    chain_label2d = ROOT.TChain("image2d_segment_tree")
    chain_label2d.AddFile('data/test_10k.root')
    chain_label2d.GetEntry(entry)
    cpp_label2d = chain_label2d.image2d_segment_branch.as_vector().front()    
    return (np.array(larcv.as_ndarray(cpp_image2d)), np.array(larcv.as_ndarray(cpp_label2d)))

image2d, label2d = get_entry(ENTRY)
fig, (ax0,ax1) = plt.subplots(1,2,figsize=(16,8), facecolor='w')
ax0.imshow(image2d, interpolation='none', cmap='jet', vmin=0, vmax=1000, origin='lower')
ax0.set_title('image',fontsize=24)
ax1.imshow(label2d, interpolation='none', cmap='jet', vmin=0, vmax=3.1, origin='lower')
ax1.set_title('label',fontsize=24)
plt.show()

input_shape  = [1,image2d.size]
image_data = np.array(image2d).reshape(input_shape)

image_dump_steps = np.concatenate((np.arange(0,100,20), 
                                   np.arange(100,400,100), 
                                   np.arange(400,1000,200), 
                                   np.arange(1000,20000,500))).astype(np.int32)

while t.current_iteration() < t.iterations():
    t.train_step()
    if t.current_iteration() in image_dump_steps:
        print('Image dump @ iteration {:d}'.format(t.current_iteration()))
        
        softmax, = t.ana(input_data = image_data)
        fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize=(24,8), facecolor='w')
        # image
        ax0.imshow(image2d, interpolation='none', cmap='jet', vmin=0, vmax=1000, origin='lower')
        ax0.set_title('image',fontsize=24)
        
        ax1.imshow(softmax[0,:,:,0], interpolation='none', cmap='jet', vmin=0, vmax=1.0, origin='lower')
        ax1.set_title('background score',fontsize=24)
        
        ax2.imshow(softmax[0].argmax(axis=2), interpolation='none', cmap='jet', vmin=0., vmax=3.1, origin='lower')
        ax2.set_title('classification', fontsize=24)
        plt.savefig('iteration_{:04d}.png'.format(t.current_iteration()))
        plt.show()
        plt.close()


