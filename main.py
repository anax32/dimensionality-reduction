# from:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# from:
# https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca
import pylab as plt

# pca
from pca import get_pca_encodings, get_pca_reconstructions

# autoencoder
from nn import (get_encodings as get_nn_encodings,
                get_reconstructions as get_nn_reconstructions)

# plot the data
def plot_encodings (x_train, x_test):
  Zenc = get_nn_encodings ("dense", x_train, x_test)
  Zpca = get_pca_encodings (x_train, x_test)

  print ("plotting...")
  plt.figure(figsize=(8,4))
  plt.subplot(121)

  print ("  pca...")
  plt.title('PCA')
  plt.scatter(Zpca[:5000,0], Zpca[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
  plt.gca().get_xaxis().set_ticklabels([])
  plt.gca().get_yaxis().set_ticklabels([])

  plt.subplot(122)

  print ("  ae...")
  plt.title('Autoencoder')
  plt.scatter(Zenc[:5000,0], Zenc[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
  plt.gca().get_xaxis().set_ticklabels([])
  plt.gca().get_yaxis().set_ticklabels([])

  plt.tight_layout()
  plt.show ()

# plot the reconstructions
def plot_reconstructions (x_train, x_test):
  Rpca = get_pca_reconstructions (x_train, x_test)
  Renc = get_nn_reconstructions ("dense", x_train, x_test)

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
  from keras.datasets import mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  print ("x_train : %s, %s" % (str (x_train.shape), str (x_train.dtype)))
  print ("x_test  : %s, %s" % (str (x_test.shape), str (x_test.dtype)))

  x_train = x_train / 255.0
  x_test = x_test / 255.0

  plot_encodings (x_train, x_test)
#  plot_reconstructions (x_train, x_test)
