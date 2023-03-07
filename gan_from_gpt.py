import torch
import torch.nn as nn
import torch.optim as optim
from rdkit.Chem import 

import zipfile
import numpy as np
import pandas as pd

# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Define the SMILES dataset
class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define the GAN model
class GAN(nn.Module):
    def __init__(self, latent_dim, output_dim, input_dim):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.generator = Generator(latent_dim, output_dim)
        self.discriminator = Discriminator(input_dim)

    def forward(self, z):
        return self.generator(z)

    def generate_samples(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        with torch.no_grad():
            samples = self.generator(z)
        return samples

    def train_step(self, data, optimizer_G, optimizer_D, criterion):
        # Train discriminator
        optimizer_D.zero_grad()
        real_data = torch.FloatTensor(data)
        real_labels = torch.ones(len(real_data), 1)
        fake_data = self.generate_samples(len(real_data)).detach()
        fake_labels = torch.zeros(len(fake_data), 1)
        d_loss_real = criterion(self.discriminator(real_data), real_labels)
        d_loss_fake = criterion(self.discriminator(fake_data), fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(len(real_data), self.latent_dim)
        fake_data = self.generator(z)
        fake_labels = torch.ones(len(fake_data), 1)
        g_loss = criterion(self.discriminator(fake_data), fake_labels)
        g_loss.backward()
        optimizer_G.step()

        return d_loss, g_loss

def train_gan(data, latent_dim, output_dim, input_dim, num_epochs, batch_size, lr, device):
    # Set up data loader
    dataset = SMILESDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up model, optimizer, and loss function
    gan = GAN(latent_dim, output_dim, input_dim).to(device)
    optimizer_G = optim.Adam(gan.generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Train the model
    for epoch in range(num_epochs):
        d_losses = []
        g_losses = []
        for batch in dataloader:
            batch = batch.to(device)
            d_loss, g_loss = gan.train_step(batch, optimizer_G, optimizer_D, criterion)
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        # Print training progress
        print(f"Epoch {epoch+1}/{num_epochs}: D_loss={torch.mean(d_losses):.4f}, G_loss={torch.mean(g_losses):.4f}")

    return gan

if __name__ == "__main__":
    zf = zipfile.ZipFile("datasets.zip", "r")
    ds = pd.read_csv(zf.open("datasets/qm9.csv"))
    
    model = train_gan(data=ds,
                      latent_dim=128,
                      output_dim=128,
                      input_dim=128)
    
    print()