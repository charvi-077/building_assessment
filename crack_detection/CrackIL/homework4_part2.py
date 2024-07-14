import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorboardX
import os

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y
    
class FashionClassifier(nn.Module):

    def __init__(self, no_hidden_layers, hidden_units,out_classes):
        super(FashionClassifier, self).__init__()
        self.input_dims = 784
        self.no_hidden_layers = no_hidden_layers
        self.hidden_units = hidden_units
        self.out_classes = out_classes
        self.hidden = nn.ModuleList()
        for i in range(self.no_hidden_layers):
            if i == 0:
                self.hidden.append(nn.Linear(self.input_dims, self.hidden_units))
                self.hidden.append(nn.BatchNorm1d(self.hidden_units))
            else:
                self.hidden.append(nn.Linear(self.hidden_units, self.hidden_units))
                self.hidden.append(nn.BatchNorm1d(self.hidden_units))
        
        self.out = nn.Linear(self.hidden_units, self.out_classes)

    
    def forward(self, x):
        for i in range(self.no_hidden_layers):
            x = F.relu(self.hidden[i](x))
        x = self.out(x)
        return x

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, writer):
    train_loss = []
    val_loss = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        print("Epoch: {}, Train Loss: {}".format(epoch, np.mean(train_loss)))
        writer.add_scalar("Loss/train", np.mean(train_loss), epoch)
    
    print("Validating...")
    model.eval()
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device).long() 
        y_hat = model(x)
        loss = criterion(y_hat, y)
        val_accuracies.append((y_hat.argmax(1) == y).float().mean().item())
        val_loss.append(loss.item())
    print("Val Loss: {}, Val Accuracy: {}".format(np.mean(val_loss), np.mean(val_accuracies)))
    writer.add_scalar("Loss/val", np.mean(val_loss))
    writer.add_scalar("Accuracy/val", np.mean(val_accuracies))
    
    return np.mean(val_accuracies)

def findBestHyperparameters(epoch_list, lr_list, l2_list, hidden_list, layer_list, batch_size_list, train_dataset, val_dataset, test_dataset, device):
    best_val_accuracy = 0
    best_hyperparameters = []
    for epoch in epoch_list:
        for lr in lr_list:
            for l2 in l2_list:
                for hidden in hidden_list:
                    for layer in layer_list:
                        for batch_size in batch_size_list:
                            
                            #dataloaders
                            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                            print("Training for epochs: {}, batch size: {}, learning rate: {}, l2 regularization: {}, hidden units: {}, hidden layers: {}".format(epoch, batch_size, lr, l2, layer, hidden))
                            
                            #model, optimizer, criterion, writer
                            model = FashionClassifier(no_hidden_layers=hidden, out_classes=out_classes, hidden_units=layer)
                            model.to(device)
                            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
                            writer = tensorboardX.SummaryWriter("runs/epochs_{}_batch_size_{}_learning_rate_{}_l2_reg_{}".format(epoch, batch_size, lr, l2))
                            
                            val_accuracy = train(model, train_loader, val_loader, criterion, optimizer, epoch, device, writer)
                            
                            if val_accuracy > best_val_accuracy:
                                print("New best val accuracy: {}".format(val_accuracy))
                                print("New best hyperparameters: {}".format([epoch, batch_size, lr, l2, hidden, layer]))
                                best_val_accuracy = val_accuracy
                                best_hyperparameters = [epoch, batch_size, lr, l2, hidden, layer]
                                best_model = model
    
    test_accuracy = testing(best_model, test_loader, device, criterion)
    
    print("Best Hyperparameters: {}".format(best_hyperparameters))
    print("Test Accuracy: {}".format(test_accuracy))

def testing(model, test_loader, device, criterion):
    model.eval()
    test_loss = []
    test_accuracies = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).long()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        test_accuracies.append((y_hat.argmax(1) == y).float().mean().item())
        test_loss.append(loss.item())
    
    print("Test Loss: {}, Test Accuracy: {}".format(np.mean(test_loss), np.mean(test_accuracies)))
    
if __name__ == "__main__":

    #loading data
    x = np.load("fashion_mnist_train_images.npy") /255. - 0.5
    y = np.load("fashion_mnist_train_labels.npy") 
    testX = np.load("fashion_mnist_test_images.npy") /255. - 0.5
    testY = np.load("fashion_mnist_test_labels.npy")

    trainX = x[:int(0.8*x.shape[0])]
    trainY = y[:int(0.8*y.shape[0])]
    valX = x[int(0.8*x.shape[0]):]
    valY = y[int(0.8*y.shape[0]):]

    train_dataset = Dataset(trainX, trainY)
    val_dataset = Dataset(valX, valY)
    test_dataset = Dataset(testX, testY)

    #defining hyperparameters and criterion
    epochs = [200,400]
    batch_size = [32,64]
    learning_rate = [0.01, 0.05]
    l2_reg = [0.01, 0.05]
    hidden_layers = [3, 5]
    hidden_units = [30, 40]
    out_classes = 10
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #finding best hyperparameters
    findBestHyperparameters(epochs, learning_rate, l2_reg, hidden_layers, hidden_units, batch_size, train_dataset, val_dataset, test_dataset, device)

    
    

    