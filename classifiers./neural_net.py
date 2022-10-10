from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4,personal_weight=None):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    if personal_weight is None:

      self.params = {}
      self.params['W1'] = std * np.random.random((input_size, hidden_size))
      self.params['b1'] = np.zeros(hidden_size)
      self.params['W2'] = std * np.random.random((hidden_size, output_size))
      self.params['b2'] = np.zeros(output_size)

    else:

      self.params = {}
      self.params = personal_weight



  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass

    first_layer = np.dot(X,W1)+b1              #(N,D)@(D,H)-->(N,H)
    act_layer = np.maximum(0,first_layer)      # ReLU Activation
    scores = np.dot(act_layer,W2)+b2           #(N,H)@(H,C)-->(N,C)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################

    # SOFTMAX LOSS -- >  L_i = exp(s_yi)/sum(exp(s_j))

    probs_mat = np.exp(scores)                           # Exponenciate every scores 

    probs_mat /= np.sum(probs_mat,axis=1).reshape(-1,1)  # Normalize Probability by dividing between the sum of each instance
    
    loss = np.zeros((N,1))
    loss = probs_mat[range(N),y]
    loss = -np.log(loss)
    loss = np.sum(loss)
    loss /= N
    loss += reg*np.sum(W1**2)+reg*np.sum(W2**2)

    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    # Grad(b2) --> -partial_b2(y_hat2)+partial_b2(exp(y_hat2))
    #              = -1+1*exp(y_hat_2) == -1+exp(scores_mat) - compress to 1 vector (1,H) 
    #--------------------------------------------------------------------------------

    aux_mat = probs_mat         # Creating aux Mat
    aux_mat[range(N),y] += -1   # Taking 1 from correct classes 
    aux_mat /= N

    grads['b2'] = np.sum(aux_mat,axis=0)  

    #--------------------------------------------------------------------------------
    # Grad(b1) --> -partial_b1(y_hat2)+partial_b1(exp(y_hat2))
    #              = W_2(-1*act_fun(y_hat1)+1*exp(y_hat_2)*act_fun(y_hat1)) 
    # 
    # W2.shape=(H,C) 
    # aux_mat.shape(N,C) 
    # first_layer.shape(N,H)

    aux_val = np.dot(aux_mat,W2.T)          #(N,C)x(C,H) = (N,H)
    aux_val *= first_layer>0                #(N,H)*(N,H) = (N,H)

    grads['b1'] = np.sum(aux_val,axis=0)    # Contract to (1,H) vector
    #---------------------------------------------------------------------------------
    #
    # Grad(W2) --> -partial_w2(y_hat2)+partial_w2(exp(y_hat2))
    #              = act_fun(y_hat1)(-1 +*exp(y_hat2))
    #
    # act_fun(y_hat1).shape = (N,H) 
    # aux_mat.shape = (N,C)

    grads['W2'] = np.dot(np.maximum(0,first_layer).T, aux_mat)  #(H,N)x(N,C)
    grads['W2'] += reg * W2 *2                              #(H,C)

    #---------------------------------------------------------------------------------

    # Grad(W1) --> -partial_w1(y_hat2)+partial_w1(exp(y_hat2))
    #              = -act_fun(y_hat1)*X+act_fun(y_hat1)*X*exp(y_hat2))
    #               = X(-act_fun(y_hat1)+act_fun(y_hat1)*exp(y_hat2))

    # W1.shape = (D,H)
    # X.shape = (N,D)
    # aux_val = (N,H)

    grads['W1'] = np.dot(X.T,aux_val)+ reg * W1*2   #(N,D).T*(N,H) = (D,H)

    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    momentum ={}

    for labels in ['W1','b1','W2','b2']:

      momentum.setdefault(labels,0) 

    for it in xrange(num_iters):
      
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass

      indx = np.random.choice(np.arange(num_train),size=batch_size) #Generate Indexes --> (0,dim)

      X_batch = X[indx,:] # Collect Sample - size ----- > (dim,batch_size)
      y_batch = y[indx]                  # Collect Sample's Classes - size ----> (batch_size,)


      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      
      old_momentum = momentum

      for label in grads:
      
        momentum[label] *= learning_rate_decay
        momentum[label] += -grads[label]*learning_rate

        self.params[label] += -learning_rate_decay*old_momentum[label]+(1+learning_rate_decay)*momentum[label]

        
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple! 
    # 
    
    first_layer = np.dot(X,self.params['W1'])+self.params['b1']              #(N,D)@(D,H)-->(N,H)
    act_layer = np.maximum(0,first_layer)      # ReLU Activation
    scores = np.dot(act_layer,self.params['W2'])+self.params['b2']           #(N,H)@(H,C)-->(N,C)

    probs = np.exp(scores)
    probs /= np.sum(scores,axis=1,keepdims=True)
    
    y_pred =np.argmax(probs,axis=1)            # pick highest high probability
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


