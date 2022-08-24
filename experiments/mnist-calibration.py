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
model.eval()

# Initialize random dataset

for _ in range(40):
    rand_tensor = torch.rand((1, 1, 32, 32))
    out = model(rand_tensor.to(device))
    label = torch.argmax(torch.softmax(out, dim=1), dim=1).item()
    print(torch.softmax(out, dim=1))