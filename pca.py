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
from keras.layers import Dense
from keras.optimizers import Adam

# init
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

print ("x_train : %s, %s" % (str (x_train.shape), str (x_train.dtype)))
print ("x_test  : %s, %s" % (str (x_test.shape), str (x_test.dtype)))

do_pca = True
do_nn = True

# pca
if do_pca:
  mu = x_train.mean(axis=0)
  U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)
  Zpca = np.dot(x_train - mu, V.transpose())

  Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
  err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
  print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

# autoencoder
if do_nn:
  m = Sequential()
  m.add(Dense(512,  activation='elu', input_shape=(784,)))
  m.add(Dense(128,  activation='elu'))
  m.add(Dense(2,    activation='linear', name="bottleneck"))
  m.add(Dense(128,  activation='elu'))
  m.add(Dense(512,  activation='elu'))
  m.add(Dense(784,  activation='sigmoid'))
  m.compile(loss='mean_squared_error', optimizer = Adam())

  history = m.fit(x_train, x_train, batch_size=128, epochs=5, verbose=1,
                  validation_data=(x_test, x_test))

  encoder = Model(m.input, m.get_layer('bottleneck').output)
  Zenc = encoder.predict(x_train)  # bottleneck representation
  Renc = m.predict(x_train)        # reconstruction

# plot the data
plt.figure(figsize=(8,4))
plt.subplot(121)
if do_pca:
  plt.title('PCA')
  plt.scatter(Zpca[:5000,0], Zpca[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
  plt.gca().get_xaxis().set_ticklabels([])
  plt.gca().get_yaxis().set_ticklabels([])

plt.subplot(122)
if do_nn:
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
