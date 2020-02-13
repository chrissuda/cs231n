import numpy as np

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
    dW =np.zeros(W.shape)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    num_class=W.shape[1]

    

    for i in range(num_train):
        scores=X[i,:].dot(W)
        scores=np.exp(scores-np.max(scores))
        score=np.sum(scores)
        si=scores[y[i]]/score  
    
        for j in range(num_class):
            sj=scores[j]/score
            dW[:,j]+=(sj-(y[i]==j))*X[i]


        loss+=-np.log(si)

    
    loss/=num_train
    dW/=num_train
    
    dW+=2*reg*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_class=W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores=X.dot(W)
    max=np.max(scores[np.arange(num_train)],axis=1)
    score=np.exp(scores-max.reshape(num_train,1))
    
    s_train_sum=np.sum(score,axis=1)
    si=score[np.arange(num_train),y]/s_train_sum
    loss=np.sum(-np.log(si))

   
    sj=score/s_train_sum.reshape(num_train,1) #reshape from (500,) to(500,1)
    sj[np.arange(num_train),y]-=1
    dW=X.T.dot(sj)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss/=num_train
    loss += reg * np.sum(W**2)
    dW/=num_train
    dW+=reg*2*W
    return loss, dW
