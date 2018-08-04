# from:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from os.path import exists, join

CACHE_FILENAME = join ("data", "models", "pca.npz")

def get_pca (pca_model_filename, x_train):
  """build a pca model from data and save to disk.
  if the file exists already, load and return that model instead
  """
  if exists (pca_model_filename):
    print ("loading model : '%s'" % pca_model_filename)
    data = np.load (pca_model_filename)
    U = data["U"]
    s = data["s"]
    V = data["V"]
  else:
    mu = x_train.mean(axis=0)
    U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)

    np.savez_compressed (pca_model_filename,
      U=U, s=s, V=V)

  return U, s, V

def get_encodings (pca_model_filename, data):
  """return the projection of data into the pca space
  """
  X = data.reshape (data.shape[0], np.prod (data.shape[1:]))
  mu = X.mean (axis = 0)

  U, s, V = get_pca (CACHE_FILENAME, X)

  return np.dot (X - mu, V.transpose ())

def get_reconstructions (data, test_data = None):
  """return reconstructions of the data after projection into the
      pca sub-space
  """
  X = data.reshape (data.shape[0], np.prod (data[1:]))

  if test_data != None:
    X_test = test_data.reshape (test_data.shape[0], np.prod (test_data[1:]))

  mu = X.mean (axis = 0)

  U, s, V = get_pca (CACHE_FILENAME, X)

  Zpca = get_pca_encodings (X, X_test)
  Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
  err = np.sum ((X-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
  print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

  return Rpca
