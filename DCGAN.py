"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a DCGAN model to generate image

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
from datetime import datetime
import torch.nn.functional as F



start_time = datetime.now()



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
Z_DIM = 100
NUM_EPOCHS = 20
FEATURES_GEN = 64
FEATURES_DISC = 64
LR = 1e-4
CHANNELS_IMG = 1
LAMBDA_GP = 10

# Dataset (without Resize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator Network 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: [batch, 100, 1, 1]
            nn.ConvTranspose2d(Z_DIM, FEATURES_GEN*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURES_GEN*8),
            nn.ReLU(),
            # [batch, 512, 4, 4]
            
            nn.ConvTranspose2d(FEATURES_GEN*8, FEATURES_GEN*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN*4),
            nn.ReLU(),
            # [batch, 256, 8, 8]
            
            nn.ConvTranspose2d(FEATURES_GEN*4, FEATURES_GEN*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN*2),
            nn.ReLU(),
            # [batch, 128, 16, 16]
            
            # Adjusted for 28x28 output
            nn.ConvTranspose2d(FEATURES_GEN*2, CHANNELS_IMG, 4, 2, 1, bias=False),
            nn.Tanh()
            # [batch, 1, 32, 32] -> We'll crop to 28x28
        )

    def forward(self, input):
        output = self.main(input)
        # Crop from 32x32 to 28x28
        return output[:, :, 2:30, 2:30]

# Discriminator Network 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            nn.Conv2d(CHANNELS_IMG, FEATURES_DISC, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # [batch, 64, 14, 14]
            
            nn.Conv2d(FEATURES_DISC, FEATURES_DISC*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(FEATURES_DISC*2),
            nn.LeakyReLU(0.2),
            # [batch, 128, 7, 7]
            
            nn.Conv2d(FEATURES_DISC*2, FEATURES_DISC*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(FEATURES_DISC*4),
            nn.LeakyReLU(0.2),
            # [batch, 256, 3, 3]
            
            nn.Conv2d(FEATURES_DISC*4, 1, 3, 1, 0, bias=False),
            # [batch, 1, 1, 1]
            nn.Flatten()  # Output: [batch, 1]
        )

    def forward(self, input):
        return self.main(input)

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Initialize models
gen = Generator().to(device)
disc = Discriminator().to(device)
gen.apply(weights_init)
disc.apply(weights_init)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

# Fixed noise
fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=device)

# Gradient Penalty function
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    
    # Check tensor sizes
    if fake_samples.size() != real_samples.size():
        fake_samples = F.interpolate(fake_samples, size=real_samples.shape[2:])
    
    alpha = torch.rand((batch_size, 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        
        # Train Discriminator
        opt_disc.zero_grad()
        
        noise = torch.randn(real.size(0), Z_DIM, 1, 1, device=device)
        fake = gen(noise)
        
        d_real = disc(real).mean()
        d_fake = disc(fake.detach()).mean()
        gp = compute_gradient_penalty(disc, real.detach(), fake.detach(), device)
        
        loss_disc = d_fake - d_real + LAMBDA_GP * gp
        loss_disc.backward()
        opt_disc.step()
        
        # Train Generator every 5 iterations
        if batch_idx % 5 == 0:
            opt_gen.zero_grad()
            fake = gen(noise)
            loss_gen = -disc(fake).mean()
            loss_gen.backward()
            opt_gen.step()
        
    # Print training stats
    print(f'Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}')
    
    # Save generated images
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu()
        save_image(fake * 0.5 + 0.5, f'./generated_images/epoch_{epoch}.png', nrow=8, normalize=True)

# Save models
torch.save(gen.state_dict(), './saved_models/generator_final.pth')
torch.save(disc.state_dict(), './saved_models/discriminator_final.pth')

end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")










