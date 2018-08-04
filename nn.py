import numpy as np
from os.path import exists

import keras
from keras.models import load_model as keras_load_model
from keras.models import Model

# config
from nn_models import (dense as dense_ae,
                       cnn as cnn_ae,
                       train as train_nn,
                       get_model_parameters)

def get_encodings (model_type, x_train, x_test):
  params = get_model_parameters (model_type, x_train, x_test)

  if exists (params["model_filename"]):
    print ("loading model : '%s'" % params["model_filename"])
    m = keras_load_model (params["model_filename"])
  else:
    m = train_nn (params)

  encoder = Model(m.input, m.get_layer('bottleneck').output)

  print ("bottleneck prediction...")
  Zenc = encoder.predict(params["x_train"])  # bottleneck representation
  print ("Zenc.shape : %s" % str (Zenc.shape))
  Zenc = Zenc.reshape ((Zenc.shape[0], Zenc.shape[-1]))

  return Zenc

def get_reconstructions (model_type, x_train, x_test):
  params = get_model_parameters (model_type, x_train, x_test)

  if exists (params["model_filename"]):
    print ("loading model : '%s'" % params["model_filename"])
    m = keras_load_model (params["model_filename"])
  else:
    m = train_nn (params)

  print ("reconstruction prediction...")
  Renc = m.predict(params["x_train"])        # reconstruction

  return Renc
