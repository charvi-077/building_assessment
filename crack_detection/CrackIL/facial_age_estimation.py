import torch
import numpy as np

#define vgg16 model with imagenet pretrained weights, and modify the last layer to output a single value
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        self.vgg16.classifier[6] = torch.nn.Linear(4096, 1)

    def forward(self, x):
        return self.vgg16(x)

def train(model, loss_fn, optimizer, batch_size, epochs, train_faces, train_ages, val_faces, val_ages):

    #train model
    for epoch in range(epochs):
        for i in range(0, len(train_faces), batch_size):
            model.train()
            batch_faces = train_faces[i:i + batch_size]
            batch_ages = train_ages[i:i + batch_size]
            predictions = model(batch_faces)
            loss = loss_fn(predictions, batch_ages)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #evaluate model on validation set
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_faces)
            val_loss = loss_fn(val_predictions, val_ages)
            print("Epoch: ", epoch, "Validation Loss: ", val_loss.item())

    #evaluate model on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_faces)
        test_loss = loss_fn(test_predictions, test_ages)
        print("Test Loss: ", test_loss.item())
if __name__ == "__main__":

    #load data
    faces = np.load('facesAndAges/faces.npy')
    ages = np.load('facesAndAges/ages.npy')

    #split data into training, validation and testing sets in 70:10:20 ratio
    train_faces = faces[:int(0.7*len(faces))]
    train_ages = ages[:int(0.7*len(ages))]
    val_faces = faces[int(0.7*len(faces)):int(0.8*len(faces))]
    val_ages = ages[int(0.7*len(ages)):int(0.8*len(ages))]
    test_faces = faces[int(0.8*len(faces)):]
    test_ages = ages[int(0.8*len(ages)):]

    #convert data to torch tensors, add 4th dimension and repeat data 3 times along the 4th dimension
    # and use bilinear interpolation to resize images to 224x224
    train_faces = torch.from_numpy(train_faces)
    train_faces = train_faces.repeat(3, 1, 1, 1).float().permute(1, 0, 2, 3)
    train_faces = torch.nn.functional.interpolate(train_faces, size=224, mode='bilinear')
    train_ages = torch.from_numpy(train_ages).float()
    val_faces = torch.from_numpy(val_faces)
    val_faces = val_faces.repeat(3, 1, 1, 1).float().permute(1, 0, 2, 3)
    val_faces = torch.nn.functional.interpolate(val_faces, size=224, mode='bilinear')
    val_ages = torch.from_numpy(val_ages).float()
    test_faces = torch.from_numpy(test_faces)
    test_faces = test_faces.repeat(3, 1, 1, 1).float().permute(1, 0, 2, 3)
    test_faces = torch.nn.functional.interpolate(test_faces, size=224, mode='bilinear')
    test_ages = torch.from_numpy(test_ages).float()

    #define model, loss function and optimizer, freeze all layers except the last one
    model = VGG16()
    model.vgg16.features.requires_grad = False
    model.vgg16.classifier.requires_grad = False
    model.vgg16.classifier[6].requires_grad = True

    #use cuda if available
    if torch.cuda.is_available():
        model = model.cuda()
        train_faces = train_faces.cuda()
        train_ages = train_ages.cuda()
        val_faces = val_faces.cuda()
        val_ages = val_ages.cuda()
        test_faces = test_faces.cuda()
        test_ages = test_ages.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 8
    epochs = 10

    #train model
    train(model, loss_fn, optimizer, batch_size, epochs, train_faces, train_ages, val_faces, val_ages)

