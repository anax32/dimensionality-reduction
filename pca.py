# from:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from os.path import exists, join

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

def get_pca_encodings (pca_model_filename, x_train, x_test):
  pca_x_train = x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  pca_x_test = x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2])
  mu = pca_x_train.mean(axis=0)

  U, s, V = get_pca (pca_model_filename, pca_x_train)

  Zpca = np.dot(pca_x_train - mu, V.transpose())

  return Zpca

def get_pca_reconstructions (x_train, x_test):
  pca_model_filename = join ("data", "models", "pca.npz")

  pca_x_train = x_train.reshape (x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  pca_x_test = x_test.reshape (x_test.shape[0], x_test.shape[1]*x_test.shape[2])
  mu = pca_x_train.mean(axis=0)

  U, s, V = get_pca (pca_model_filename, pca_x_train)

  Zpca = get_pca_encodings (x_train, x_test)
  Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
  err = np.sum((pca_x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
  print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

  return Rpca
