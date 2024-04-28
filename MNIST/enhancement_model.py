import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils_mnist import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load a complex autoencoder
autoencoder = ComplexAutoencoder()
autoencoder.to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
criterion = nn.MSELoss()
# load emnist data
transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
emnist_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, transform=transform, download=True)
dataloader = DataLoader(emnist_dataset, batch_size=64, shuffle=True)

# train the complex autoencoder
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    autoencoder.train()
    for data in dataloader:
        inputs = data[0].to(device)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        outputs=(outputs-0.5)*2
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')
torch.save(autoencoder.state_dict(), "autoencoder_model.pth")
