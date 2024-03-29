from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', cache=False)
X_dataset = mnist.data.astype('float32').reshape(-1, 1, 28, 28)/255
y_dataset = mnist.target.astype('int64')
XCnn, XCnnt, y, yt = train_test_split(X_dataset, y_dataset, test_size=1/7, ) #random_state=42)

def plot_example(X, y):
    """Plot the first 5 images and their labels in a row."""
    for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, 28), y[:5])):
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(y)
###plot_example(X_train, y_train)


import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X_dataset.shape[3]*X_dataset.shape[3]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(mnist.target))

class Cnn(nn.Module):
    def __init__(self, dropout=0.5):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(100, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        
        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

"""
The below bit of code takes a lot of time to compute
It can be faster by reducing the "max_epochs" to a lower number (still greater than 0)
In other hand, by reducing the "max_epochs", this method loses accuracy
In order to make a quick run in this code, I did set "max_epochs" to 1 (althought a value arrond 15 is more desirable)
"""
from skorch import NeuralNetClassifier

torch.manual_seed(0)
net = NeuralNetClassifier(
    Cnn,
    max_epochs=1,
    lr=0.1,
    device=device,
)
net.fit(XCnn, y);


from sklearn.metrics import accuracy_score

Predictions_Cnn = net.predict(XCnnt)
Cnn_accuracy = accuracy_score(yt, Predictions_Cnn)

print('Cnn classification accuracy = {}%'.format(100*Cnn_accuracy))
