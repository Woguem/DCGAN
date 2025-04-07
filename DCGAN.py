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

start_time = datetime.now()  # Start timer


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
Z_DIM = 100  # Size of noise vector
NUM_EPOCHS = 50
FEATURES_GEN = 64  # Number of feature maps in generator
FEATURES_DISC = 64  # Number of feature maps in discriminator
LR = 2e-4  # Learning rate
CHANNELS_IMG = 1

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load dataset 
dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,  
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z_DIM-dimensional noise
            nn.ConvTranspose2d(Z_DIM, FEATURES_GEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 8),
            nn.ReLU(),
            
            # State size: (FEATURES_GEN*8) x 4 x 4
            nn.ConvTranspose2d(FEATURES_GEN * 8, FEATURES_GEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 4),
            nn.ReLU(),
            
            # State size: (FEATURES_GEN*4) x 8 x 8
            nn.ConvTranspose2d(FEATURES_GEN * 4, FEATURES_GEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 2),
            nn.ReLU(),
            
            # State size: (FEATURES_GEN*2) x 16 x 16
            nn.ConvTranspose2d(FEATURES_GEN * 2, FEATURES_GEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN),
            nn.ReLU(),
            
            # State size: (FEATURES_GEN) x 32 x 32
            nn.ConvTranspose2d(FEATURES_GEN, CHANNELS_IMG, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: CHANNELS_IMG x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is CHANNELS_IMG x 32 x 32
            nn.Conv2d(CHANNELS_IMG, FEATURES_DISC, 4, 2, 1, bias=False),
            nn.ReLU(),
            
            # State size: FEATURES_DISC x 16 x 16
            nn.Conv2d(FEATURES_DISC, FEATURES_DISC * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC * 2),
            nn.ReLU(),
            
            # State size: (FEATURES_DISC*2) x 8 x 8
            nn.Conv2d(FEATURES_DISC * 2, FEATURES_DISC * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC * 4),
            nn.ReLU(),
            
            # State size: (FEATURES_DISC*4) x 4 x 4
            nn.Conv2d(FEATURES_DISC * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Initialize models
gen = Generator().to(device)
disc = Discriminator().to(device)


# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LR)
opt_disc = optim.Adam(disc.parameters(), lr=LR)

# Loss function
criterion = nn.BCELoss()

# Fixed noise for visualization
fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=device)

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
        
        # Train Discriminator
        opt_disc.zero_grad()
        
        # Real images
        output_real = disc(real).view(-1)
        loss_disc_real = criterion(output_real, torch.ones_like(output_real))
        
        # Fake images
        fake = gen(noise).detach()
        output_fake = disc(fake).view(-1)
        loss_disc_fake = criterion(output_fake, torch.zeros_like(output_fake))
        
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc.backward()
        opt_disc.step()
        
        # Train Generator
        opt_gen.zero_grad()
        fake = gen(noise)
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        loss_gen.backward()
        opt_gen.step()
        
        # Print training stats
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}')
    
    # Save generated images
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu()
        save_image(fake * 0.5 + 0.5, f'./generated_images/epoch_{epoch}.png', nrow=8, normalize=True)


# Save final model weights
torch.save(gen.state_dict(), './saved_models/generator_final.pth')
torch.save(disc.state_dict(), './saved_models/discriminator_final.pth')

end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")










