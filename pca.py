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

#x_train = x_train.reshape(60000, 784) / 255.0
#x_test = x_test.reshape(10000, 784) / 255.0

print ("x_train : %s, %s" % (str (x_train.shape), str (x_train.dtype)))
print ("x_test  : %s, %s" % (str (x_test.shape), str (x_test.dtype)))

# pca
if do_pca:
  pca_x_train = x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  pca_x_test = x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2])
  mu = pca_x_train.mean(axis=0)

  if exists (pca_model_filename):
    data = np.load (pca_model_filename)
    U = data["U"]
    s = data["s"]
    V = data["V"]
  else:
    U,s,V = np.linalg.svd(pca_x_train - mu, full_matrices=False)

    np.savez_compressed (pca_model_filename,
      U=U, s=s, V=V)

  Zpca = np.dot(pca_x_train - mu, V.transpose())
  Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
  err = np.sum((pca_x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
  print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

# autoencoder
if do_nn:
  cnn_x_train = x_train[...,np.newaxis]
  cnn_x_test = x_test[...,np.newaxis]

  def dense_ae (input_shape):
    from keras.layers import Dense
    m = Sequential()
    m.add(Dense(512,  activation='elu', input_shape=input_shape))
    m.add(Dense(128,  activation='elu'))
    m.add(Dense(2,    activation='linear', name="bottleneck"))
    m.add(Dense(128,  activation='elu'))
    m.add(Dense(512,  activation='elu'))
    m.add(Dense(784,  activation='sigmoid'))
    print (m.summary ())
    return m

  def cnn_ae (input_shape):
    from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D
    from keras.regularizers import l1

    m = Sequential ()
    m.add (Conv2D (64, (3,3), strides=(2,2), activation="elu", input_shape=input_shape))
    m.add (Conv2D (32, (3,3), strides=(2,2), activation="elu"))
    m.add (Conv2D (16,  (3,3), activation="elu"))
    m.add (Conv2D (2,   (3,3), strides=(2,2), activation="linear", name="bottleneck", activity_regularizer=l1(10e-5)))
    m.add (Conv2DTranspose (8, (4,4), strides=(1,1), activation="elu"))
    m.add (Conv2DTranspose (16, (3,3), strides=(1,1), activation="elu"))
    m.add (Conv2DTranspose (16, (3,3), strides=(2,2), activation="elu"))
    m.add (Conv2DTranspose (16, (3,3), strides=(2,2), activation="elu"))
    m.add (ZeroPadding2D ( ((0,1),(0,1)) ) )
    m.add (Conv2D (1, (1,1), activation="sigmoid"))
    print (m.summary ())
    return m

#  m = dense_ae ((784,))
  if exists (nn_model_filename):
    print ("loading model : '%s'" % nn_model_filename)
    m = keras.models.load_model (nn_model_filename)
  else:
    m = cnn_ae (cnn_x_train.shape[1:])
    m.compile(loss='mean_squared_error', optimizer = Adam())

    history = m.fit(cnn_x_train, cnn_x_train,
                    batch_size=16,
                    epochs=5,
                    verbose=1,
                    validation_data=(cnn_x_test, cnn_x_test))
    m.save (nn_model_filename)

  encoder = Model(m.input, m.get_layer('bottleneck').output)

  print ("bottleneck prediction...")
  Zenc = encoder.predict(cnn_x_train)  # bottleneck representation
  print ("reconstruction prediction...")
  Renc = m.predict(cnn_x_train)        # reconstruction

print ("Zenc.shape : %s" % str (Zenc.shape))
Zenc = Zenc.reshape ((Zenc.shape[0], Zenc.shape[-1]))

# plot the data
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
