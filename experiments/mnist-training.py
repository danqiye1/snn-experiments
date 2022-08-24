"""
Testing the calibration of MNIST classifiers and their relationship to 
"""
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Pad
from patterns.models import LeNet
from patterns.utils import validate, train_epoch

from pdb import set_trace as bp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize and load the pre-trained model
model = LeNet()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize dataset
transforms = Compose([ToTensor(), Pad(2)])
trainset = torchvision.datasets.MNIST(
                root="/mnt/hdd/data", 
                download=True, 
                train=True,
                transform=transforms)
trainloader = DataLoader(
                    trainset,
                    batch_size=16,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4
                )
testset = torchvision.datasets.MNIST(
                root="/mnt/hdd/data", 
                download=True,
                transform=transforms)
testloader = DataLoader(
                    testset,
                    batch_size=16,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4
                )
epochs = 40
best_verror = 1_000_000

for epoch in tqdm(range(epochs)):
    loss = train_epoch(model, trainloader, optimizer=optimizer, criterion=criterion, device=device)
    vloss, verror = validate(model, testloader, criterion=criterion, device=device)

    tqdm.write(f"Epoch: {epoch}, Training loss: {loss:.3f}, Val loss: {vloss:.3f}, Val Error: {verror: .3f}")

    if verror < best_verror:
        torch.save(model.state_dict(), "best-model.pth")
        best_verror = verror




