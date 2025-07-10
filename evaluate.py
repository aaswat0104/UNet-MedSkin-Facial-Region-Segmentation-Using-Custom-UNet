import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from model.unet import UNet

# Paths
image_dir = 'data/images'
model_path = 'checkpoints/unet_medskin.pth'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Run Evaluation
image_files = os.listdir(image_dir)
for file in image_files:
    img_path = os.path.join(image_dir, file)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0][0].cpu().numpy()

    # Save prediction
    plt.imsave(os.path.join(output_dir, f"mask_{file}"), pred, cmap='gray')

    # Optional: show side-by-side
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[1].imshow(pred, cmap='gray')
    ax[1].set_title("Predicted Mask")
    plt.show()
