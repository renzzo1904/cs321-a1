from copy import copy
from turtle import shape
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0

  ## Since the Gradient can be obatined with \partial F / \partial W we see that analitically:

  # dF/dW = d/dW [1/N * sum(L_i)+sum (W_k^2)] = dF/dW = 1/N* d/dW[sum (max(0,W*x-(W*x)'+1)) + d/dW [sum(W^2)] ]  --- > +-x + 2*sum(W)

  for i in xrange(num_train):

    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):

      
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if margin > 0:

        loss += margin

        dW[:,y[i]] -= X[i]/num_train  # We take the EW difference of rows that are the good label for the instance
        dW[:,j]    += X[i]/num_train  # We add EW the pixels of the image that is incorrect

        dW[i,j] +=2*reg*W[i,j]

      

    #############################################################################
    # TODO:         
    #                                                             #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.    
    #                                    
    #############################################################################


  loss /= num_train
  loss += reg*np.sum(W**2)

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_instances = X.shape[0]
  num_classes = W.shape[1]

  scores_mat = X.dot(W)

  c_mat = np.zeros_like(scores_mat)
  ic_mat = np.ones_like(scores_mat)
  correct_mat_grad = np.zeros_like(X)

  for x in xrange(num_instances):

    c_mat[x,y[x]] += 1
    ic_mat[x,y[x]] -= 1

  L_i_mat = np.maximum(0,ic_mat*scores_mat-ic_mat*np.sum(scores_mat*c_mat,axis=1).reshape(-1,1)+1*ic_mat)

  loss = np.sum(L_i_mat)/num_instances + reg*np.sum(W**2)

  # ic_mat_sum = np.sum(ic_mat,axis=0).reshape(1,-1)          # (1,num_clases) containing number of times that the class has been incorrect 
  # c_mat_sum = np.sum(c_mat,axis=0).reshape(1,-1)       # (1,num_clases) containing number of times that the class has been correct
  # X_add = X.T.dot(np.ones((num_instances,1)))   # (num_features,1) containing sum of each pixel over all instances

  
  aux_array = np.zeros_like(L_i_mat)                    # Defining firt aux matrix
  aux_array [L_i_mat>0] = 1                             # Places where loss is greater than zero indicate incorrect classes or positive loss.
  rw_sum = aux_array.sum(axis=1)                        # Summing over all the columns, to end up with (Num_instances) 
  aux_array[np.arange(num_instances),y] = -rw_sum       # Substracting this for the specific positions of correct classes.

  dW = X.T.dot(aux_array)
  dW /= num_instances
  dW += 2*reg*W

    # Now aux_array (num_instances, num_classes) is a matrix with ones accounting for the incorrect classes, 
    # which add and not reapeat over the sumation in one loss instance L_i. And also the number of times that the specific column of pixels 
    # had to be substracted in positive Loss. (Negative Loss dows not exist and that is why the number isn´t constant)
    
    # When know multiplied by X.T (num_pixels, num_instances) by this aux_array (num_instances,num_classes) we allign all the pixels #1,#2 and so on 
    # with the class 1instance columns. Then for example: pix1_i1*ni1(number of times its been used in that first instance e.g its the correct pixel in that instance?) + 
    # pix1_i2*ni2+pix1_i3*ni3 +...
  

    
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
