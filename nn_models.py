# from:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np

import keras
from keras.models import Model

# config
from os.path import exists, join

def dense (input_shape):
  """build a model made of dense layers"""
  from keras.layers import Input, Reshape, Dense
  from keras.regularizers import l2
  from keras.constraints import unit_norm

  dense_params = {
    "activation" : "elu"
  }

  bottleneck_params = {
    "name" : "bottleneck",
    "activation" : "linear"
  }

  input = x = Input (shape=input_shape)
  x = Dense(512, kernel_constraint=unit_norm (), **dense_params) (x)
  x = Dense(128, kernel_constraint=unit_norm (), **dense_params) (x)
  x = Dense(2,   kernel_constraint=unit_norm (), **bottleneck_params) (x)
  x = Dense(128, **dense_params) (x)
  x = Dense(512, **dense_params) (x)
  output = x = Dense(784,  activation='sigmoid') (x)
  return [input], [output]

def rigged_dense (input_shape, search_params):
  """build a model made of dense layers which takes
     hypteropt search parameters.
     search parameters are:
       "activation": activation function
       "kernel_constraint": kernel constraint function
  """
  from keras.layers import Input, Reshape, Dense
  from keras.regularizers import l2
  from keras.constraints import unit_norm

  dense_params = {
    "activation" : search_params["activation_fn"],
    "kernel_constraint" : search_params["kernel_constraint"] or None
  }

  bottleneck_params = {
    "name" : "bottleneck",
    "activation" : "linear"
  }

  input = x = Input (shape=input_shape)
  x = Dense(512, **dense_params) (x)
  x = Dense(128, **dense_params) (x)
  x = Dense(2,   **bottleneck_params) (x)
  x = Dense(128, **dense_params) (x)
  x = Dense(512, **dense_params) (x)
  output = x = Dense(784,  activation='sigmoid') (x)
  return [input], [output]

def cnn (input_shape):
  """build a convolutional neural net based autoencoder
  """
  from keras.layers import Input, Conv2D, Conv2DTranspose, ZeroPadding2D
  from keras.regularizers import l2
  from keras.constraints import max_norm

  conv_params = {
    "kernel_size" : (3, 3),
    "activation" : "elu",
    "kernel_constraint" : max_norm (3)
  }

  bottleneck_params = dict (conv_params)
  bottleneck_params["activation"] = "linear"
  bottleneck_params["strides"] = (2, 2)
  bottleneck_params["name"] = "bottleneck"

  input = x = Input (shape=input_shape)
  x = Conv2D (128, strides=(2,2), **conv_params) (x)
  x = Conv2D (64, strides=(2,2), **conv_params) (x)
  x = Conv2D (32, **conv_params) (x)
  x = Conv2D (2, **bottleneck_params) (x)
  x = Conv2DTranspose (8, (4,4), strides=(1,1), activation="elu") (x)
  x = Conv2DTranspose (16, (3,3), strides=(1,1), activation="elu") (x)
  x = Conv2DTranspose (32, (3,3), strides=(2,2), activation="elu") (x)
  x = Conv2DTranspose (64, (3,3), strides=(2,2), activation="elu") (x)
  x = ZeroPadding2D (((0,1),(0,1))) (x)
  output = x = Conv2D (1, (1,1), activation="sigmoid") (x)
  return [input], [output]

def get_model_parameters (model_type, x_train, x_test):
  """build a dict of model hyperparameters for a requested
     model_type.
  """
  from hyperopt import hp

  dense_params = {
    "x_train" : x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2]),
    "x_test" : x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2]),
    "batch_size" : 128,
    "epochs" : 50,
    "input_shape" : (x_train.shape[1]*x_train.shape[2], ),
    "constructor_fn" : lambda : dense ((x_train.shape[1]*x_train.shape[2],)),
    "loss" : "mean_squared_error",
    "optimizer" : "adam",
    "model_filename" : join ("data", "models", "dense_constraint_unit_norm.h5")
  }

  rigged_dense_params = {
    "rigged" : True,

    "x_train" : x_train.reshape (x_train.shape[0], np.prod (x_train.shape[1:])),
    "x_test" : x_test.reshape (x_test.shape[0], np.prod (x_test.shape[1:])),
    "constructor_fn" : lambda params: rigged_dense (np.prod (x_train.shape[1:], params)),
    "model_filename" : join ("data", "models", "optimal_dense_constraint_unit_norm.h5"),

    "batch_size" : hp.choice ("@batch_size", [16, 32, 64, 128, 256]),
    "epochs" : hp.choice ("@epochs", [10, 20, 50, 100]),
    "loss" : hp.choice ("@loss", ["mae", "mse", "binary_crossentropy"]),
    "optimizer" : hp.choice ("@optimizer", ["adam", "sgd", "rmsprop"])
  }

  cnn_params = {
    "x_train" : x_train,
    "x_test" : x_test,
    "batch_size" : 16,
    "epochs" : 5,
    "input_shape" : x_train.shape[1:],
    "constructor_fn" : lambda : cnn ((x_train.shape[1:])),
    "loss" : "mean_squared_error", # "binary_crossentropy"
    "optimizer" : "adam", # "adadelta", "sgd"
    "model_filename" : join ("data", "models", "cnn.h5")
  }

  if model_type == "dense":
    return dense_params
  elif model_type == "dense_rigged":
    return rigged_dense_params
  elif model_type == "cnn":
    return cnn_params
  else:
    raise NotImplementedException ("Unknown model_type : '%s'" % model_type)
