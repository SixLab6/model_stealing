import time
import random
import torchvision
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data
from utils_mnist import *

torch.manual_seed(0)
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# hyperparameter
dataroot = "data"
workers = 0
batch_size = 100
image_size = 28
nc = 1
num_classes = 10
nz = 100
ngf = 64
ndf = 64
num_epochs = 10
lr = 0.0003
beta1 = 0.5
ngpu = 1
# get the training set (some samples from EMNIST and some samples from MNIST)
my_transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.EMNIST(
    root=dataroot,
    split='digits',
    train=True,
    transform=my_transform,
    download=True
)
test_data = datasets.EMNIST(
    root=dataroot,
    split='digits',
    train=False,
    transform=my_transform
)
my_data,mylabel=[],[]
for idx, (data, target) in enumerate(train_data):
    if idx>99999:
        break
    data[0]=torch.rot90(torch.flipud(data[0]),k=-1)
    my_data.append(data)
    mylabel.append(target)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=my_transform)
# select the size of subset
subset_size = 20000
total_samples = len(trainset)
subset_indices = random.sample(range(total_samples), subset_size)
subset_dataset = torch.utils.data.Subset(trainset, subset_indices)
for i,(data,target) in enumerate(subset_dataset):
    my_data.append(data)
    mylabel.append(target)
    if i>20000:
        break
mylabel = torch.LongTensor(mylabel)
my_data = torch.FloatTensor(torch.stack(my_data))
torch_dataset = Data.TensorDataset(my_data, mylabel)
print(f'Total Size of Dataset: {len(torch_dataset)}')
dataloader = DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)

# Create the generator
netG = Generator(ngpu).to(device)
# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)
# Create the Discriminator
netD = Discriminator(ngpu).to(device)
# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netD.apply(weights_init)
# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label_num = 1.
fake_label_num = 0.
# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# Label one-hot for G
label_1hots = torch.zeros(10,10)
for i in range(10):
    label_1hots[i,i] = 1
label_1hots = label_1hots.view(10,10,1,1).to(device)
# Label one-hot for D
label_fills = torch.zeros(10, 10, image_size, image_size)
ones = torch.ones(image_size, image_size)
for i in range(10):
    label_fills[i][i] = ones
label_fills = label_fills.to(device)

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
loss_tep = 10

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    beg_time = time.time()
    netG.train()
    netG.cuda()
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        # Create batch of latent vectors and laebls that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(100, nz, 1, 1).to(device)
        fixed_label = label_1hots[torch.arange(10).repeat(10).sort().values]
        netD.zero_grad()
        # Format batch
        real_image = data[0].to(device)
        b_size = real_image.size(0)
        real_label = torch.full((b_size,), real_label_num).to(device)
        fake_label = torch.full((b_size,), fake_label_num).to(device)
        G_label = label_1hots[data[1]]
        D_label = label_fills[data[1]]
        # Forward pass real batch through D
        output = netD(real_image, D_label).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, real_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1).to(device)
        # Generate fake image batch with G
        fake = netG(noise, G_label)
        # Classify all fake batch with D
        output = netD(fake.detach(), D_label).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, fake_label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, D_label).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, real_label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        # Output training stats
        end_time = time.time()
        run_time = round(end_time - beg_time)
        print(
            f'Epoch: [{epoch + 1:0>{len(str(num_epochs))}}/{num_epochs}]',
            f'Step: [{i + 1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
            f'Loss-D: {errD.item():.4f}',
            f'Loss-G: {errG.item():.4f}',
            f'D(x): {D_x:.4f}',
            f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
            f'Time: {run_time}s',
            end='\r'
        )

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Save D(X) and D(G(z)) for plotting later
        D_x_list.append(D_x)
        D_z_list.append(D_G_z2)

        # Save the Best Model
        if errG < loss_tep:
            torch.save(netG.state_dict(), 'model.pt')
            loss_tep = errG
    # Check how the generator is doing by saving G's output on fixed_noise and fixed_label
    with torch.no_grad():
        fake = netG(fixed_noise, fixed_label).detach().cpu()
    img_list.append(utils.make_grid(fake, nrow=10))

    print()

    netG.eval()
    netG.cpu()
    # Generate the Fake Images
    with torch.no_grad():
        fake = netG(fixed_noise.cpu(), fixed_label.cpu())
    # Plot the fake images
    plt.axis("off")
    plt.title("Fake Images")
    fake = utils.make_grid(fake, nrow=10)
    plt.imshow(fake.permute(1, 2, 0) * 0.5 + 0.5)
    plt.show()

torch.save(netG.state_dict(), 'last_model.pt')
