import numpy as np

def train_rigged_model (search_params):
  from keras.models import Model
  from keras.datasets import mnist
  from keras.layers import Input, Reshape, Dense
  from keras.regularizers import l2
  from keras.constraints import unit_norm, max_norm
  from keras.callbacks import EarlyStopping
  from hyperopt import STATUS_OK

  print (search_params)

  input_shape = (28*28, )

  dense_params = {
    "activation" : search_params["activation_fn"]
  }

  constrained_dense_params = dict (dense_params)
  constrained_dense_params["kernel_constraint"] = search_params["kernel_constraint"][1]

  bottleneck_params = dict (constrained_dense_params)
  bottleneck_params["name"] = "bottleneck"
  bottleneck_params["activation"] = "linear"

  input = x = Input (shape=input_shape)
  x = Dense(512, **constrained_dense_params) (x)
  x = Dense(128, **constrained_dense_params) (x)
  x = Dense(2,   **bottleneck_params) (x)
  x = Dense(128, **dense_params) (x)
  x = Dense(512, **dense_params) (x)
  output = x = Dense(784,  activation='sigmoid') (x)

  m = Model (inputs=[input], outputs=[output])

  loss = search_params["loss"]
  optimizer = search_params["optimizer"]
  batch_size = search_params["batch_size"]
  epochs = search_params["epochs"]

  m.compile (
    loss = search_params["loss"],
    optimizer = search_params["optimizer"]
  )

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape ((x_train.shape[0], np.prod (x_train.shape[1:])))
  y_train = x_train.reshape ((x_train.shape[0], np.prod (x_train.shape[1:])))

  x_test = x_test.reshape ((x_test.shape[0], np.prod (x_test.shape[1:])))
  y_test = x_test.reshape ((x_test.shape[0], np.prod (x_test.shape[1:])))

  print ("x_train:shape '%s'" % str (x_train.shape))
  print ("x_test:shape  '%s'" % str (x_test.shape))

  x_train = x_train / 255.0
  y_train = y_train / 255.0
  x_test = x_test / 255.0
  y_test = y_test / 255.0

  history = m.fit (
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    shuffle = True,
    validation_data = (x_test, y_test),
    callbacks = [
      EarlyStopping (min_delta=0.001)
    ]
  )

  score = m.evaluate (x_test, y_test, verbose = 0)
  accuracy = score
  return {"loss" : -accuracy, "status" : STATUS_OK, "model": m}

def train_optimal_model (x_train, y_train, x_test, y_test, model_params):
  from os.path import join
  import keras
  from hyperopt import tpe

  print ("train_optimal_model")

  best = fmin (
    fn = train_rigged_model,
    space = {
      "activation_fn" : hp.choice ("@activation_fn",
      [
        "elu", "relu", "sigmoid", "linear"
      ]),

      "kernel_constraint" : hp.choice ("@kernel_constraint",
      [
        ("none", None),
        ("unit_norm", keras.constraints.unit_norm ()),
        ("max_norm(1)", keras.constraints.max_norm (1)),
        ("max_norm(2)", keras.constraints.max_norm (2)),
        ("max_norm(3)", keras.constraints.max_norm (3))
      ]),

      "loss" : hp.choice ("@loss", ["mean_squared_error", "mean_absolute_error", "binary_crossentropy"]),
      "optimizer" : hp.choice ("@optimizer", ["adam", "sgd", "rmsprop"]),
      "batch_size" : hp.choice ("@batch_size", [16, 32, 64, 128, 256]),
      "epochs" : hp.choice ("@epochs", [10, 20, 50, 100]),
    },
    algo = tpe.suggest,
    max_evals = 10
  )

  print (best)

  best_model.save (join ("data", "models", "optimal.h5"))
