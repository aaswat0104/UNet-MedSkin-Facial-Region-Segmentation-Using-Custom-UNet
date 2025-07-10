import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.unet import UNet
from losses.custom_loss import DiceBCELoss
from utils import SegmentationDataset
from torchvision import transforms
import os

# -----------------------------
# ✅ Paths
image_dir = 'data/images'
mask_dir = 'data/masks'

# -----------------------------
# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# -----------------------------
# ✅ Dataset & DataLoader
dataset = SegmentationDataset(image_dir, mask_dir, transform, mask_transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# ✅ Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)  # 🔧 FIXED LINE HERE
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# ✅ Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        output = model(img)
        loss = criterion(output, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

# -----------------------------
# ✅ Save model
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/unet_medskin.pth')
