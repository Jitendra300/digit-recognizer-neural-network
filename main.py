import math
import argparse
import numpy as np
from sklearn.datasets import load_digits

# Creating the parser for command arguments
parser = argparse.ArgumentParser(description='Malware Analysis Program')
# Add arguments
parser.add_argument('--batch_size', type=int, default=8, help='The batch size of the training dataset. Default = 8')
parser.add_argument('--learning_rate', type=float, default=0.03, help='Learning rate (alpha) of our model. Default = 0.03')
parser.add_argument('--epoch', type=int, default=65, help='Epoch size. Default = 65')
parser.add_argument('--datasplit', type=int, default=1000, help='Data split between testing and training. Default = 1000. Nearly 60/40 distribution.')

# Parse the arguments
args = parser.parse_args()


# creating minibatches
def miniBatch(A, batchSize):
    numObs = A.shape[0]
    batches = []
    batchNum = math.floor(numObs / batchSize)

    if(numObs%batchSize == 0):
        for i in range(batchNum):
            aBatch = A[i * batchSize:(i + 1) * batchSize, :]
            batches.append(aBatch)
    else:
        for i in range(batchNum):
            aBatch = A[i * batchSize:(i + 1) * batchSize, :]
            batches.append(aBatch)
        aBatch = A[batchNum * batchSize:, :]
        batches.append(aBatch)
    return batches

# this is onehotencoding
def onehotencoding(arr):
    ans =[]
    for i in arr:
        tempArr = []
        for j in range(10):
            if(i == j):
                tempArr.append(1)
            else :
                tempArr.append(0)
        ans.append(tempArr)
    return ans

# weights and biases set to random
def initializingParameters():
    np.random.seed(1)
    w1 = np.random.rand(10, 64)
    w2 = np.random.rand(10, 10)
    b1 = np.random.rand(10, 1)
    b2 = np.random.rand(10, 1)
    return w1, w2, b1, b2

def ReLu(Z):
    return np.maximum(0, Z)

def ReLu_prime(Z):
    return Z > 0

def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))

def forwardPropagation(w1, w2, b1, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    return z1, z2, a1, a2

def backwardPropagation(w1,w2, z1, a1, a2, X, Y):
    size = Y.size
    proposationNum = 1/size

    dz2 = a2 - Y
    dw2 = proposationNum * np.dot(dz2, a1.T)
    db2 = proposationNum * np.sum(dz2)

    dz1 = np.dot(w2.T, dz2) * ReLu_prime(z1)
    dw1 = proposationNum * np.dot(dz1, X.T)
    db1 = proposationNum * np.sum(dz1)

    return dw1, dw2, db1, db2

def updateParams(w1, w2, b1, b2, dw1, dw2, db1, db2, learningRate):
    w1 = w1  - learningRate*dw1
    w2 = w2  - learningRate*dw2
    b1 = b1 - learningRate*db1
    b2 = b2 - learningRate*db2
    return w1, w2, b1, b2

def predictionAccuracy(w1, w2, b1, b2, xTest, yTest):
    prediction = forwardPropagation(w1, w2, b1, b2, xTest.T)[-1]
    score = 0
    totalScore = 0
    for i in range(len(prediction.T)):
        if(np.argmax(prediction.T[i]) == np.argmax(yTest[i])):
            score+=1
        totalScore+=1
    print("Accuracy: ",(100*score/totalScore))

def gradientDescient(X, Y, epoch, learningRate):
    w1, w2, b1, b2 = initializingParameters()
    for i in range(epoch):
        for j in range(len(X)):
            # taking transpose of it so that dimensions match while dot product
            current_X_training_dataset = np.array(X[j]).T
            current_Y_training_dataset = np.array(Y[j]).T
            z1, z2, a1, a2 = forwardPropagation(w1, w2, b1, b2, current_X_training_dataset)
            dw1, dw2, db1, db2 = backwardPropagation(w1, w2,z1, a1, a2, current_X_training_dataset, current_Y_training_dataset)
            w1, w2, b1, b2 = updateParams(w1, w2, b1, b2, dw1, dw2, db1, db2, learningRate)
        # checks how good our  model is doing for every iteration of 100
        print("Epoch : ", i)
        predictionAccuracy(w1, w2, b1, b2, xTest, yTest)
    return w1, w2, b1, b2


# Variables are declared here...
learningRate = args.learning_rate # also known as alpha
dataSplit = args.datasplit # this indicates that first 1000 would be trainingdata and other would be testdata
batchSize = args.batch_size
epoch = args.epoch

"""
this includes data of images and labels
where each image is of a single digit number which has dimensions 8x8,
and label of each image gives the digit number.
 """

data = load_digits()
imagesData = data["data"]
labelData = data["target"]

# Training data
whole_xTrain = imagesData[:dataSplit]
whole_yTrain = np.array(onehotencoding(labelData[:dataSplit]))

#  Testing data
xTest = imagesData[dataSplit:]
yTest  = np.array(onehotencoding(labelData[dataSplit:]))

xTrain = miniBatch(whole_xTrain, batchSize)
yTrain = miniBatch(whole_yTrain, batchSize)

w1, w2, b1, b2 = gradientDescient(xTrain, yTrain, epoch, learningRate)

print("------------------------------------\n")
print("Final Results: ")
print("Testing on Test Datasets: ")
predictionAccuracy(w1, w2, b1, b2, xTest, yTest)
