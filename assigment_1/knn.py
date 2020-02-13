import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from collections import Counter
class NearestNeighbor():
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in tqdm(range(num_test)):
            # find the nearest training image to the i'th test image
            # using  L1 distance (sum of absolute value differences)
            #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            #using L2 distance
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
        return Ypred

(Xtr, Ytr),(Xte, Yte)=cifar10.load_data()
Ytr=Ytr.flatten()
Yte=Yte.flatten()
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.astype('float32').reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.astype('float32').reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)


print('accuracy: %2.2f' % (( np.mean(Yte_predict == Yte))*100),"%")

# plt.plot(x[0:100],Yte_predict[0:100],'ro',color='blue',label='Ytrain',ms=1)
# plt.plot(x[0:100],Yte[0:100],'ro',color='red' ,label='Ytest',ms=1)
# plt.title('Model accuracy')
# plt.ylabel('Y')
# plt.xlabel('Epoch')
# plt.legend(['Yte_predict', 'Yte'], loc='upper left')
# plt.show()
# print(Counter(Yte))