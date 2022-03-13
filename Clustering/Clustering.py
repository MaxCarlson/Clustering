import torch
import numpy as np
import torchvision
from torch import nn
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision import datasets, models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
#device = torch.device('cpu')
device = torch.device(dev) 

transform = transforms.Compose([transforms.ToTensor()])

batchSize = 128

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

indicies = []
imagesPerCatagory = 1000
catagoriesCount = {x : 0 for x in range(10)}
for i in range(len(trainset)):
    if catagoriesCount[trainset[i][1]] < imagesPerCatagory:
        catagoriesCount[trainset[i][1]] += 1
        indicies.append(i)
    
    filledCatagories = 0
    for k, v in catagoriesCount.items():
        if v == imagesPerCatagory:
            filledCatagories += 1
        else:
            break
    if filledCatagories == 10:
        break

batchSize = 128
subsetSize = imagesPerCatagory*10
subset = torch.utils.data.Subset(trainset, indicies)
trainloader = torch.utils.data.DataLoader(subset, batch_size=batchSize, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=2)

kmeansloader = torch.utils.data.DataLoader(subset, batch_size=subsetSize)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

def make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)


def kmeans(x, y, reshapeSize):
    model = KMeans(n_clusters=10)
    x = np.reshape(x, (subsetSize, reshapeSize))
    predicted_labels = model.fit_predict(x)


    cm = confusion_matrix(y, predicted_labels)
    indexes = linear_sum_assignment(make_cost_m(cm))
    indexes = np.transpose(np.asarray(indexes))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    ax = sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
    plt.show()
    accuracy = np.trace(cm2) / np.sum(cm2)
    print(f'Accuracy = {accuracy}')

def kmeansfvectors():
    def train(model, num_epochs=5, learning_rate=1e-3):
        torch.manual_seed(42)
        criterion = nn.MSELoss() # mean square error loss
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate, 
                                     weight_decay=1e-5) # <--
        outputs = []
        for epoch in range(num_epochs):
            for data in trainloader:
                img, _ = data
                recon = model(img)
                loss = criterion(recon, img)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            for data in testloader:
                img, _ = data
                recon = model(img)
                tloss = criterion(recon, img)

            print('Epoch:{}, Traing Loss:{:.4f}, Test Loss:{:.4f}'.format(epoch+1, float(loss), float(tloss)))

    model = Autoencoder()
    max_epochs = 20
    train(model, num_epochs=max_epochs)

    x = next(iter(kmeansloader))
    x, y = x[0].numpy(), x[1].numpy()
    #x = np.reshape(x, (subsetSize, 28*28))
    x = torch.tensor(x)
    out = model.encode(x)
    #print(out)

    kmeans(out.detach().numpy(), y, 64)

    a=5

if __name__ == '__main__':
    #x = next(iter(kmeansloader))
    #x, y = x[0].numpy(), x[1].numpy()
    #kmeans(x, y, 28*28)
    kmeansfvectors()

