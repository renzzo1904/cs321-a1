
import numpy as np
from random import shuffle
from past.builtins import xrange

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

  dW = np.zeros_like(W)
  num_instances, num_features = X.shape
  num_classes = W.shape[1]
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  scores = X.dot(W) # (num_instances, num_clases)
  probs_mat= np.zeros_like(scores)
  loss = np.zeros((num_instances,1))
  
  #  HAVE IN MIND
  # - Shape of Dw = (num_features, num_classes)
  # - Shape of Loss = (num_features,1)
  # - Shape of Probs_mat = (num_instances,num_classes)

  # For gradient we have to know that we have to add all the features with them peers (eg. pix#1 is paired with all the other pix#1s of the other instances)

  probs_mat = np.exp(scores)

  for i in xrange(num_instances):

    probs_mat[i,:] /= np.sum(probs_mat[i,:])

    correct_class = y[i]

    for m in xrange(num_features):
      for j in xrange(num_classes):

        if j == correct_class:
          dW[m,j] += X.T[m,i]*(-1+probs_mat[i,j])
        else:
          dW[m,j] += X.T[m,i]*(probs_mat[i,j])
          
    loss[i] = probs_mat[i,correct_class]

  loss = np.sum(-np.log(loss))
  loss /= num_instances
  loss += 0.5 * reg * np.sum(W**2)

  dW /= num_instances
  dW += reg*W
  
    

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
  
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  num_instances = X.shape[0]
  scores = X.dot(W)                                    # get scores 

  probs_mat = np.exp(scores)                           # Exponenciate every scores

  probs_mat /= np.sum(probs_mat,axis=1).reshape(-1,1)  # Normalize Probability by dividing between the sum of each instance
  
  loss = np.zeros((num_instances,1))
  loss = probs_mat[range(num_instances),y]
  loss = -np.log(loss)
  loss = np.sum(loss)
  loss /= num_instances
  loss += 0.5*reg*np.sum(W**2)

  aux_probs_mat = probs_mat
  aux_probs_mat[range(num_instances), y] -= 1    # Adds the auxiliar -1 needed for correct classes
  dW = X.T.dot(probs_mat)                        #(num_features,num_instances)@(num_instances,num_classes)
  dW /= num_instances
  dW += reg*W

  # LOGIC of ABOVE OPERATION:

  # If we get all the pixel #1 from all instances and multiply them by the normalized probability of getting that specific probability of 
  # belonging to class 1,2, etc in that specific instance. That goes on for pixel 1,2,3 and so on until filling (num_features,num_classes)



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  
  return loss, dW

