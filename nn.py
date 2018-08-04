import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from os.path import exists

from keras.models import load_model as keras_load_model

from nn_models import get_model_parameters
from nn_train import train as train_nn
from nn_optimal import (train_optimal_model as train_optimal_nn)

def get_encodings (model_type, x_train, x_test):
  from keras.models import Model

  params = get_model_parameters (model_type, x_train, x_test)

  if params is None:
    m = train_optimal_nn (x_train, x_train, x_test, x_test, params)
  else:
    if exists (params["model_filename"]):
      print ("loading model : '%s'" % params["model_filename"])
      m = keras_load_model (params["model_filename"])
    else:
      m = train_nn (params)

  encoder = Model(m.input, m.get_layer('bottleneck').output)

  print ("bottleneck prediction...")
  if params is None:
    Zenc = encoder.predict (x_train)
  else:
    Zenc = encoder.predict(params["x_train"])  # bottleneck representation

  print ("Zenc.shape : %s" % str (Zenc.shape))
  return Zenc.reshape ((Zenc.shape[0], Zenc.shape[-1]))

def get_reconstructions (model_type, x_train, x_test, count):
  params = get_model_parameters (model_type, x_train, x_test)

  if exists (params["model_filename"]):
    print ("loading model : '%s'" % params["model_filename"])
    m = keras_load_model (params["model_filename"])
  else:
    m = train_nn (params)

  print ("reconstruction prediction...")
  Renc = m.predict(params["x_train"][:count,...])        # reconstruction

  return Renc


if __name__ == "__main__":
  import sys
  from keras.datasets import mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data ()

#  model_types = ["dense", "cnn", "dense_rigged", "cnn_rigged"]
#
#  if sys.argv[1] in model_types:
#    params = get_model_parameters (sys.argv[1], x_train, x_test)
#    m = train_nn (params)
  train_optimal_nn (x_train, x_train, x_test, x_test)
