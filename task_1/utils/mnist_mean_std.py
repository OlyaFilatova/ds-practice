"""Calculate and print the mean and standard deviation of the MNIST dataset."""
from torchvision import datasets, transforms
import torch

train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
data = next(iter(loader))
images, labels = data

mean = images.mean().item()
std = images.std().item()
print(mean, std)
