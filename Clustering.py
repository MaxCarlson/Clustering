import torch
import random
import numpy as np
import torchvision
from torch import nn
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

def addNoise(x):
    _, row, col = x.shape
    number_of_pixels = int(28*28*3/8)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        x[0][y_coord][x_coord] = 1

    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        x[0][y_coord][x_coord] = 0
    return x

transform = transforms.Compose([transforms.ToTensor()])
noisyTransform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Lambda(addNoise)])

batchSize = 128

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

noisyTrainset = datasets.MNIST(root='./data', train=True, download=True, transform=noisyTransform)
noisyTestset = datasets.MNIST(root='./data', train=False, download=True, transform=noisyTransform)

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

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return xA, xB
    
    def __len__(self):
        return len(self.datasetA)

batchSize = 128
subsetSize = imagesPerCatagory*10
trainsubset = torch.utils.data.Subset(trainset, indicies)
trainNoisySubset = torch.utils.data.Subset(noisyTrainset, indicies)


trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=batchSize, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=2)

noisyTrainSet = NoisyDataset(trainNoisySubset, trainsubset)
noisyTestSet = NoisyDataset(noisyTestset, testset)
noisyTrainloader = torch.utils.data.DataLoader(noisyTrainSet, batch_size=batchSize, shuffle=True, num_workers=2)
noisyTestloader = torch.utils.data.DataLoader(noisyTestSet, batch_size=batchSize, shuffle=True, num_workers=2)


kmeansloader = torch.utils.data.DataLoader(trainsubset, batch_size=subsetSize)

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

def trainAE(model, num_epochs, trainloader, testloader, learning_rate=1e-3,):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=1e-5) # <--
    outputs = []
    for epoch in range(num_epochs):
        for img, _ in trainloader:
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for img, _ in testloader:
            recon = model(img)
            tloss = criterion(recon, img)

        print('Epoch:{}, Traing Loss:{:.4f}, Test Loss:{:.4f}'.format(epoch+1, float(loss), float(tloss)))


def kmeansfvectors(doPca = False):
    max_epochs = 20
    model = Autoencoder()
    trainAE(model, max_epochs, trainloader, testloader)

    x = next(iter(kmeansloader))
    x, y = x[0].numpy(), x[1].numpy()
    #x = np.reshape(x, (subsetSize, 28*28))
    x = torch.tensor(x)
    out = model.encode(x).detach().numpy()
    #print(out)

    outSize = 64

    if doPca:
        pcCount = 4
        out = PCA(n_components=pcCount).fit_transform(np.reshape(out, (out.shape[0], outSize)))
        outSize = pcCount

    kmeans(out, y, outSize)

def trainNoisyAE(model, num_epochs, trainloader, testloader, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=1e-5) # <--
    outputs = []
    for epoch in range(num_epochs):
        #if epoch % 5 == 0 and epoch != 0:
        #    fig, axs = plt.subplots(1, 3)
        #    axs[0].imshow(oimg[0][0])
        #    axs[1].imshow(img[0][0])
        #    axs[2].imshow(recon.detach().numpy()[0][0])
        #    plt.show()
        for (img, _), (oimg, _) in trainloader:
            if dev == 'cuda:0':
                img = img.to(device)
                oimg = oimg.to(device)

            recon = model(img)
            loss = criterion(recon, oimg)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #for (img, _), (oimg, _) in testloader:
        #    recon = model(img)
        #    tloss = criterion(recon, oimg)

        #print('Epoch:{}, Traing Loss:{:.4f}, Test Loss:{:.4f}'.format(epoch+1, float(loss), float(tloss)))
        print('Epoch:{}, Traing Loss:{:.4f}'.format(epoch+1, float(loss)))


    fig, axs = plt.subplots(3, 10)
    for (img, _), (oimg, _) in testloader:
        if dev == 'cuda:0':
            img = img.to(device)

        recon = model(img)

        if dev == 'cuda:0':
            recon = recon.cpu()
            img = img.cpu()
        recon = recon.detach().numpy()

        for i in range(10):
            axs[0][i].imshow(oimg[i][0])
            axs[1][i].imshow(img[i][0])
            axs[2][i].imshow(recon[i][0])
            axs[0][i].axis('off')
            axs[1][i].axis('off')
            axs[2][i].axis('off')

        plt.show()

def noisyAutoencoder():
    max_epochs = 15
    model = Autoencoder()
    model = model.to(device)
    trainNoisyAE(model, max_epochs, noisyTrainloader, noisyTestloader)

if __name__ == '__main__':
    #x = next(iter(kmeansloader))
    #x, y = x[0].numpy(), x[1].numpy()
    #kmeans(x, y, 28*28)
    #kmeansfvectors(doPca=False)
    #kmeansfvectors(doPca=True)
    noisyAutoencoder()

