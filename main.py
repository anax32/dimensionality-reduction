# from:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# from:
# https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca
import pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.optimizers import Adam

# config
from os.path import exists
do_pca = True
do_nn = True
nn_model_filename = "nn_reg.h5"
pca_model_filename = "pca.npz"

# init
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print ("x_train : %s, %s" % (str (x_train.shape), str (x_train.dtype)))
print ("x_test  : %s, %s" % (str (x_test.shape), str (x_test.dtype)))

x_train = x_train / 255.0
x_test = x_test / 255.0

# pca
def get_pca (pca_model_filename, x_train):
  if exists (pca_model_filename):
    data = np.load (pca_model_filename)
    U = data["U"]
    s = data["s"]
    V = data["V"]
  else:
    U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)

    np.savez_compressed (pca_model_filename,
      U=U, s=s, V=V)

  return U, s, V

def get_pca_encodings (x_train, x_test):
  pca_x_train = x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  pca_x_test = x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2])
  mu = pca_x_train.mean(axis=0)

  U, s, V = get_pca (pca_model_filename, pca_x_train)

  Zpca = np.dot(pca_x_train - mu, V.transpose())

  return Zpca

def get_pca_reconstructions (x_train, x_test):
  pca_x_train = x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  pca_x_test = x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2])
  mu = pca_x_train.mean(axis=0)

  U, s, V = get_pca (pca_model_filename, pca_x_train)

  Zpca = get_pca_encodings (x_train, x_test)
  Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
  err = np.sum((pca_x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
  print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

  return Rpca

# autoencoder
if do_nn:
  cnn_x_train = x_train[...,np.newaxis]
  cnn_x_test = x_test[...,np.newaxis]
  dense_x_train = x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  dense_x_test = x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2])

  def dense_ae (input_shape):
    from keras.layers import Input, Reshape, Dense
    from keras.regularizers import l2
    from keras.constraints import max_norm

    dense_params = {
      "activation" : "elu"
    }

    bottleneck_params = {
      "name" : "bottleneck",
      "activation" : "linear"
    }

    input = x = Input (shape=input_shape)
    x = Dense(512, kernel_constraint=max_norm (784*512), **dense_params) (x)
    x = Dense(128, kernel_constraint=max_norm (512*128), **dense_params) (x)
    x = Dense(2,   kernel_constraint=max_norm (128*2), **bottleneck_params) (x)
    x = Dense(128, **dense_params) (x)
    x = Dense(512, **dense_params) (x)
    output = x = Dense(784,  activation='sigmoid') (x)
    return [input], [output]

  def cnn_ae (input_shape):
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

  def train_nn (params):
    inputs, outputs = params["constructor_fn"] ()

    m = Model (inputs=inputs, outputs=outputs)

    print (m.summary ())

    m.compile(loss=params["loss"], optimizer = params["optimizer"])

    history = m.fit(params["x_train"], params["x_train"],
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    verbose=1,
                    shuffle=True,
                    validation_data=(params["x_test"], params["x_test"]))
    m.save (params["model_name"])

    return m

dense_params = {
  "x_train" : x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2]),
  "x_test" : x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2]),
  "batch_size" : 128,
  "epochs" : 50,
  "input_shape" : (x_train.shape[1]*x_train.shape[2], ),
  "constructor_fn" : lambda : dense_ae ((x_train.shape[1]*x_train.shape[2],)),
  "loss" : "mean_squared_error",
  "optimizer" : "adam",
  "model_filename" : nn_model_filename
}

cnn_params = {
  "x_train" : x_train,
  "x_test" : x_test,
  "batch_size" : 16,
  "epochs" : 5,
  "input_shape" : x_train.shape[1:],
  "constructor_fn" : lambda : cnn_ae ((x_train.shape[1:])),
  "loss" : "mean_squared_error", # "binary_crossentropy"
  "optimizer" : "adam", # "adadelta", "sgd"
  "model_filename" : nn_model_filename
}

def get_nn_encodings (nn_model_type):
  if nn_model_type == "dense":
    params = dense_params
  elif nn_model_type == "cnn":
    params = cnn_params
  else:
    raise NotImplementedException ("Unknown model type '%s'" % nn_model_type)

  if exists (params["model_filename"]):
    print ("loading model : '%s'" % params["model_filename"])
    m = keras.models.load_model (params["model_filename"])
  else:
    m = train_nn (params)

  encoder = Model(m.input, m.get_layer('bottleneck').output)

  print ("bottleneck prediction...")
  Zenc = encoder.predict(params["x_train"])  # bottleneck representation
  print ("Zenc.shape : %s" % str (Zenc.shape))
  Zenc = Zenc.reshape ((Zenc.shape[0], Zenc.shape[-1]))

  return Zenc

def get_nn_reconstructions (nn_model_type):
  if nn_model_type == "dense":
    params = dense_params
  elif nn_model_type == "cnn":
    params = cnn_params
  else:
    raise NotImplementedException ("Unknown model type '%s'" % nn_model_type)

  if exists (params["model_filename"]):
    print ("loading model : '%s'" % params["model_filename"])
    m = keras.models.load_model (params["model_filename"])
  else:
    m = train_nn (params)

  print ("reconstruction prediction...")
  Renc = m.predict(params["x_train"])        # reconstruction

  return Renc

# plot the data
def plot_encodings ():
  Zenc = get_nn_encodings ("dense")
  Zpca = get_pca_encodings (x_train, x_test)

  print ("plotting...")
  plt.figure(figsize=(8,4))
  plt.subplot(121)
  if do_pca:
    print ("  pca...")
    plt.title('PCA')
    plt.scatter(Zpca[:5000,0], Zpca[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

  plt.subplot(122)
  if do_nn:
    print ("  ae...")
    plt.title('Autoencoder')
    plt.scatter(Zenc[:5000,0], Zenc[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

  plt.tight_layout()
  plt.show ()

# plot the reconstructions
def plot_reconstructions ():
  Rpca = get_pca_reconstructions (x_train, x_test)
  Renc = get_nn_reconstructions ("dense")

  plt.figure(figsize=(9,3))
  toPlot = (x_train, Rpca, Renc)
  for i in range(10):
    for j in range(3):
      ax = plt.subplot(3, 10, 10*j+i+1)
      plt.imshow(toPlot[j][i,:].reshape(28,28),
                 interpolation="nearest",
                 vmin=0, vmax=1)
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

  plt.tight_layout()
  plt.show ()

# execute
if __name__ == "__main__":
  plot_encodings ()
#  plot_reconstructions ()
