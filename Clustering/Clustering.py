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

indicies = []
imagesPerCatagory = 100
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

batchSize=imagesPerCatagory*10
subset = torch.utils.data.Subset(trainset, indicies)
trainloader = torch.utils.data.DataLoader(subset, batch_size=batchSize, shuffle=True, num_workers=2)


def make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

if __name__ == '__main__':
    model = KMeans(n_clusters=10)
    x = next(iter(trainloader))
    x, y = x[0].numpy(), x[1].numpy()
    x = np.reshape(x, (batchSize, 28*28))
    predicted_labels = model.fit_predict(x)


    cm = confusion_matrix(y, predicted_labels)
    indexes = linear_sum_assignment(make_cost_m(cm))
    indexes = np.asarray(indexes)
    indexes = np.transpose(indexes)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    ax = sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
    #plt.show()
    accuracy = np.trace(cm2) / np.sum(cm2)
    print(f'Accuracy = {accuracy}')

