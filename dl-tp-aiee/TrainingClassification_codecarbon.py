# Import packages 
import numpy as np
import os
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from codecarbon import EmissionsTracker


# Load MNIST dataset
train_data = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_data = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the model
class Digits(nn.Module):
    def __init__(self):
        super(Digits, self).__init__()
        self.hidden_layer = nn.Linear(28*28, 32)
        self.output_layer = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.sigmoid(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Define device where to run the training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Digits().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create Carbon tracker
tracker = EmissionsTracker()
#if offline, it is possible to specify measurement specifications
#tracker = OfflineEmissionsTracker(country_iso_code='SGP', cloud_provider='gcp',  cloud_region='asia-southeast1', log_level='error')

tracker.start()
# Running the training loop
for epoch in range(5):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0
    
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)
    
    epoch_train_loss = train_loss / len(train_data)
    epoch_train_acc = train_acc / len(train_data)
    epoch_val_loss = val_loss / len(test_data)
    epoch_val_acc = val_acc / len(test_data)
    
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epoch+1, 5, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))


emissions: float = tracker.stop()
print('-----------------------------------------------------')
print('Total CPU energy consumption CodeCarbon (Process): ' + str(tracker._total_cpu_energy.kWh*1000) + ' Wh')
print('Total RAM energy consumption CodeCarbon (Process): ' + str(tracker._total_ram_energy.kWh*1000) + ' Wh')
print('Total GPU energy consumption CodeCarbon (Process): ' + str(tracker._total_gpu_energy.kWh*1000) + ' Wh')
print('Total Energy consumption CodeCarbon (Process): ' + str(tracker._total_energy.kWh*1000) + ' Wh')
print('Emissions by CodeCarbon (Process): '+ str(emissions*1000) + ' gCO2e')