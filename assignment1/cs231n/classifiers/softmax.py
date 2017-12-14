import numpy as np
from random import shuffle
# from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D = X.shape
  C = W.shape[1]

  for i in range(N):
    s = X[i,:].dot(W)
    exp_s = np.exp(s - s.max())
    p = exp_s / exp_s.sum()
    loss += -np.log(p[y[i]])
    p[y[i]] -= 1
    dW += np.dot(X[i, :].reshape((D, 1)), p.reshape((1, C))) 

  loss /= N
  dW /= N

  loss += np.sum(W ** 2)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  s = X.dot(W)
  s_max = s.max(axis=1).reshape((N, 1))
  exp_s = np.exp(s - s_max) 

  i = np.arange(X.shape[0])
  p = exp_s / exp_s.sum(axis=1, keepdims=True)
  loss = -np.log(p[i, y]).mean() + reg * np.sum(W ** 2)

  j_eq_y = np.zeros_like(p)
  j_eq_y[i, y] = 1.0
  dW = X.T.dot(p - j_eq_y) / N + reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW