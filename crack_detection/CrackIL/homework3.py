import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def softmax_cross_entropy(y_pred, y, w_1, alpha):

    n = y.shape[0]
    CE_loss = -np.sum(y * np.log(y_pred)) / n
    reg_loss =  alpha * (np.sum(w_1**2))
    loss = CE_loss + reg_loss
    # print("CE Loss:", CE_loss)
    # print("Reg Loss:", reg_loss)
    return loss
def backward_pass(x, y_pred, y, w_1, b_1, lr, alpha):

    n = y.shape[0]
    w_1_grad = (x.T @ (y_pred - y)) / n + 2 * alpha * w_1
    b_1_grad = np.sum(y_pred - y, axis=0, keepdims=True) / n
    w_1 = w_1 - lr * w_1_grad
    b_1 = b_1 - lr * b_1_grad
    return w_1, b_1

def forward_pass(x, w_1, b_1):

    z_1 = x @ w_1 + b_1
    y_pred = softmax(z_1)
    return y_pred

def train(x_tr, y_tr, w_1_init, b_1_init, epochs, batch_size, lr, alpha):

    w_1 = w_1_init
    b_1 = b_1_init
    for i in range(epochs):
        epoch_loss = 0
        for j in range(0, x_tr.shape[0], batch_size):
            x_batch = x_tr[j:j+batch_size]
            y_batch = y_tr[j:j+batch_size]
            y_pred = forward_pass(x_batch, w_1, b_1)
            loss = softmax_cross_entropy(y_pred, y_batch, w_1, alpha)
            epoch_loss += loss
            w_1, b_1 = backward_pass(x_batch, y_pred, y_batch, w_1, b_1, lr, alpha)
        print("Epoch: ", i, "Total Epoch Loss: ", epoch_loss)
    
    return w_1, b_1

def validation(w_1, b_1, x_val, y_val):

    y_pred = forward_pass(x_val, w_1, b_1)
    loss = softmax_cross_entropy_unreg(y_pred, y_val)
    return loss, y_pred

def softmax_cross_entropy_unreg(y_pred, y):

    n = y.shape[0]
    loss = -np.sum(y * np.log(y_pred)) / n
    return loss

def accuracy(y_pred, y):

    n = y.shape[0]
    correct = 0
    for i in range(n):
        if np.argmax(y_pred[i]) == np.argmax(y[i]):
            correct += 1
    return correct / n

if __name__ == "__main__":
    
    x = np.load('fashion_mnist_train_images.npy')
    y = np.load('fashion_mnist_train_labels.npy')
    x_te = np.load('fashion_mnist_test_images.npy')
    y_te = np.load('fashion_mnist_test_labels.npy')
    
    #transform y_tr and y_te to one-hot encoding
    y = np.eye(10)[y]
    y_te = np.eye(10)[y_te]
    
    #split the training data into training and validation data, 80% for training and 20% for validation
    x_tr = x[:int(0.8*x.shape[0])]
    y_tr = y[:int(0.8*y.shape[0])]
    x_val = x[int(0.8*x.shape[0]):]
    y_val = y[int(0.8*y.shape[0]):]

    #define a 2 layer softmax neural network
    #input layer: 784 nodes
    #output layer: 10 nodes

    #reshape the input data
    x_tr = x_tr.reshape(-1,784)
    x_te = x_te.reshape(-1,784)

    #initialize weights and biases
    w1_init = np.random.randn(784,10)/np.sqrt(784)
    b1_init = np.random.randn(1,10)
    

    #define hyperparameters
    # epochs = 800
    # batch_size = [16, 32, 64, 128]
    # lr = [0.00002, 0.00001, 0.000005, 0.000001]
    # alpha = [0.0001, 0.0005, 0.00001, 0.00005]
    epochs = 800
    batch_size = 16
    lr = 0.000001
    alpha = 0.0001
    # min_validation_cost = np.inf
    # max_accuracy = 0
    #train the network
    w, bias = train(x_tr, y_tr, w1_init, b1_init, epochs, batch_size, lr, alpha)
    cost,y_pred = validation(w, bias, x_val, y_val)
    acc = accuracy(y_pred, y_val)
    w_min = w
    bias_min = bias
    print(cost, acc)
    # for e in epochs:
    #     for b in batch_size:
    #         for l in lr:
    #             for a in alpha:
    #                 w, bias = train(x_tr, y_tr, w1_init, b1_init, e, b, l, a)
    #                 cost,y_pred = validation(w, bias, x_val, y_val)
    #                 acc = accuracy(y_pred, y_val)
    #                 print('Epochs:, Batch Size:, Learning Rate:, Alpha:',(e,b,l,a))
    #                 print('Validation Loss:',cost)
    #                 print('Accuracy:',acc)

    #                 if cost<min_validation_cost and acc>max_accuracy:
    #                     print("Found improved parameters")
    #                     print("New min validation cost:",cost)
    #                     print("New max accuracy:",acc)
    #                     print("Optimal Parameters: Epochs:, Batch Size:, Learning Rate:, Alpha:",(e,b,l,a))

    #                     min_validation_cost = cost
    #                     max_accuracy = acc
    #                     w_min = w
    #                     bias_min = bias

    test_loss, test_pred = validation(w_min, bias_min, x_te, y_te)
    test_acc = accuracy(test_pred, y_te)
    print('Test Loss:',test_loss)
    print('Test Accuracy:', test_acc)
