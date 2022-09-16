import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 32

use_cuda = True
print('Cuda Available:', torch.cuda.is_available())
device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')

# Load all the data set
train_data = np.load('../dataSet/train_set.npy', allow_pickle=True)
train_data = train_data.astype(float)
train_data = train_data/255.
# print('train_data shape:', train_data.shape)
val_data = np.load('../dataSet/test_set.npy', allow_pickle=True)
val_data = val_data.astype(float)
val_data = val_data/255.
# print('val_data shape:', val_data.shape)

train_labels = np.load('../dataSet/train_labels.npy', allow_pickle=True)
train_labels = train_labels.astype(int)
# print('train_labels shape:', train_labels.shape)

val_labels = np.load('../dataSet/test_labels.npy', allow_pickle=True)
val_labels = val_labels.astype(int)
# print('val_labels shape:', val_labels.shape)


val_labels = val_labels.astype(int)

classes = ('Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Local', 'Random', 'Scratch', 'Near-Full', 'None')

def batchify_data(x_data, y_data, batch_size):
    N = int(len(y_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({'x': torch.tensor(x_data[i:i+batch_size], dtype=torch.float32),
                        'y': torch.tensor(y_data[i:i+batch_size], dtype=torch.int64)})
    return batches

def visualize(data, label):
    # The input data.shape is (3,64,64), need to transpose it
    data = np.transpose(data, (1, 2, 0))
    plt.imshow(data)
    plt.show()
    print(classes[label])

# Batchify data
train_data = batchify_data(train_data, train_labels, batch_size)
val_data = batchify_data(val_data, val_labels, batch_size)
# visualize(np.array(train_data[1]['x'][25]), int(train_data[1]['y'][25]))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.flatten = Flatten()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(2, 2),
            # nn.Dropout(0.2),

            self.flatten,
            nn.Dropout(),
            nn.Linear(2048,512),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(2048),
            # nn.Dropout(0.2),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(2048),
            #nn.Dropout(0.2),
            nn.Linear(512,9),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.cpu().numpy(), y.cpu().numpy()))


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x,y
        inputs, labels = batch['x'], batch['y']

        # print(inputs)
        # print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get output prediction
        outputs = model(inputs)

        # Predict and store accuracy
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, labels))

        # Compute losses
        loss = F.nll_loss(outputs, labels)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


def train_model(train_data, dev_data, model, lr=0.0001, momentum=0.9, nesterov=False, n_epochs=1):
    """Train a model for N epochs given data and hyper-params."""
    # We optimize with SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train | loss: {:.6f}  accuracy: {:.6f}'.format(loss, acc))
        losses.append(loss)
        accuracies.append(acc)

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
        print('Valid | loss: {:.6f}  accuracy: {:.6f}'.format(val_loss, val_acc))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save model
        path = './vgg.pth'
        torch.save(model.state_dict(), path)

    return losses,accuracies,val_losses,val_accuracies


if __name__ == '__main__':
    model = VGG16().to(device)

    # train_model(train_data,val_data,model)

