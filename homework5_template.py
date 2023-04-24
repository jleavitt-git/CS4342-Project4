import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

IM_WIDTH = 48
NUM_INPUT = IM_WIDTH**2
NUM_HIDDEN = 20
NUM_OUTPUT = 1

def relu (z):
    return z * (z > 0)

def reluDerivative(z):
    return 1 * (z > 0)

def forward_prop (x, y, W1, b1, W2, b2):
    z = np.dot(W1, x) + b1[:, np.newaxis]
    h = relu(z)
    yhat = np.dot(W2, h) + b2
    loss = (1/(2*y.size)) * (np.sum((y - yhat) ** 2))
    #print(f"------Stats------\nW1 = {W1.shape}\nW2 = {W2.shape}\nb1 = {b1.shape}\nb2 = {b2}\nx = {x.shape}\ny = {y}\nyhat = {yhat}\nz = {z.shape}\nh = {h.shape}\nloss = {loss}")
    return loss, x, z, h, yhat

def back_prop (X, y, W1, b1, W2, b2):
    loss, x, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2)
    #print (y.size)
    gT1 = np.dot(np.transpose(yhat-y), W2)
    #print (gT1.shape)
    gT2 = reluDerivative(z.T)
    #print(f"------Stats------\nW1 = {W1.shape}\nW2 = {W2.shape}\nb1 = {b1.shape}\nb2 = {b2}\nx = {x.shape}\ny = {y}\nyhat = {yhat}")
    gT = gT1 * gT2
    gradW2 = np.dot(yhat-y, h.T)
    gradb2 = np.mean(yhat-y, axis = 1) #average across all training examples
    gradW1 = np.dot(gT.T, x.T)
    gradb1 = np.mean(gT.T, axis = 1) #average across all training examples
    return gradW1, gradb1, gradW2, gradb2

def train (trainX, trainY, W1, b1, W2, b2, testX, testY, epsilon = 1e-5, batchSize = 128, numEpochs = 250):
    #print (trainX.shape)
    #print (trainY.shape)
    for i in range (numEpochs):
        if i <= (numEpochs*.1):
            epsilon = 1e-4
        elif i <= (numEpochs*.5):
            epsilon = 1e-5
        else: epsilon = 1e-6
        if i % 500 == 0 and i != 0:
            print(f"Progress {i/50}%")
        for j in range((int(trainY.size/batchSize)) - 1):
            gradW1, gradb1, gradW2, gradb2 = back_prop(trainX[:, j*batchSize:(j*batchSize) + batchSize], trainY[j*batchSize:(j*batchSize) + batchSize], W1, b1, W2, b2)
            W1 = W1-(epsilon*gradW1)
            W2 = W2-(epsilon*gradW2)
            b1 = b1-(epsilon*gradb1)
            b2 = b2-(epsilon*gradb2)
        #print(f"------Stats------\nW1 = {W1}\nW2 = {W2}\nb1 = {b1}\nb2 = {b2}")
    return W1, b1, W2, b2

def show_weight_vectors (W1):
    # Show weight vectors in groups of 5.
    for i in range(NUM_HIDDEN//5):
        plt.imshow(np.hstack([ np.pad(np.reshape(W1[idx,:], [ IM_WIDTH, IM_WIDTH ]), 2, mode='constant') for idx in range(i*5, (i+1)*5) ]), cmap='gray'), plt.show()
    plt.show()

def loadData (which, mu = None):
    images = np.load("age_regression_X{}.npy".format(which)).reshape(-1, 48**2).T
    labels = np.load("age_regression_y{}.npy".format(which))

    if which == "tr":
        mu = np.mean(images)

    # TODO: you may wish to perform data augmentation (e.g., left-right flipping, adding Gaussian noise).

    return images - mu, labels, mu

def checkGradient():
    testW1 = np.load("testW1.npy")
    testb1 = np.load("testb1.npy")
    testW2 = np.load("testW2.npy")
    testb2 = np.load("testb2.npy")
    oneSampleX = np.load("oneSampleX.npy")
    oneSampley = np.load("oneSampley.npy")
    gradW1, gradb1, gradW2, gradb2 = back_prop(np.atleast_2d(oneSampleX).T, oneSampley, testW1, testb1, testW2, testb2)
    correctGradW1 = np.load("correctGradW1OnSample.npy")
    correctGradb1 = np.load("correctGradb1OnSample.npy")
    correctGradW2 = np.load("correctGradW2OnSample.npy")
    correctGradb2 = np.load("correctGradb2OnSample.npy")
    # The differences should all be <1e-5
    print(np.sum(np.abs(gradW1 - correctGradW1)))
    print(np.sum(np.abs(gradb1 - correctGradb1)))
    print(np.sum(np.abs(gradW2 - correctGradW2)))
    print(np.sum(np.abs(gradb2 - correctGradb2)))

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY, mu = loadData("tr")
        testX, testY, _ = loadData("te", mu)

    # Check the gradient value for correctness.
    # Note: the gradients shown below assume 20 hidden units.
    checkGradient()

    import csv
    with open ('HyperParamAccuracies.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['nodes', 'epochs', 'batchSize', 'LossTrain', 'LossTest'])
        nodes = [5, 10, 20, 30]
        epochs = [50, 100, 250, 500]
        batchSizes = [32, 64, 128, 256]
        for node in nodes:
            for ep in epochs:
                for bs in batchSizes:
                    # Initialize weights to reasonable random values
                    W1 = 2*(np.random.random(size=(node, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
                    b1 = 0.01 * np.ones(node)
                    W2 = 2*(np.random.random(size=(NUM_OUTPUT, node))/node**0.5) - 1./node**0.5
                    b2 = np.mean(trainY)
                    # Train NN
                    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, testX, testY, batchSize = bs, numEpochs=ep)
                    #Evaluate Accuracy
                    loss, x, z, h, yhat = forward_prop(trainX, trainY, W1, b1, W2, b2)
                    lossTest, xTest, zTest, hTest, yhatTest = forward_prop(testX, testY, W1, b1, W2, b2)
                    print (f"Train Accuracy: nodes = {node}, epochs = {ep}, batchSize = {bs}, Loss = {loss}")
                    print (f"Test Accuracy: nodes = {node}, epochs = {ep}, batchSize = {bs}, Loss = {lossTest}")
                    writer.writerow([node, ep, bs, loss, lossTest])
                    #show_weight_vectors(W1)
