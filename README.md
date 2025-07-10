# UNet-MedSkin-Facial-Region-Segmentation-Using-Custom-UNet
To design a custom UNet architecture that performs facial skin segmentation from RGB face images. This project serves as a foundational POC in the broader context of medical/cosmetic imaging and generative vision systems.

🧠 Model Highlights

- Architecture: Custom UNet with dynamic depth, skip connections, and batch normalisation
- Input Size: 512 × 512 RGB images
- Output: 1-channel binary mask (same resolution)
- Loss Function: Combined Dice Loss + Binary Cross Entropy
- Training Backend: PyTorch (GPU-compatible)
- Evaluation: Visual side-by-side comparisons of original images and predicted masks

outputs::

![image](https://github.com/user-attachments/assets/4b2a3a34-721f-4b30-8124-b72b3ca04d45)
![image](https://github.com/user-attachments/assets/14ebadb7-6c62-4127-8278-35f858d7e519)

🧪 Evaluation Results

| Metric | Value (example) |
|--------|-----------------|
| Average Loss | 0.5958 after 10 epochs |
| Dice Score (est.) | ~0.78 (visually accurate on small set) |
| Training Time | ~2 mins (6-image dataset) |

🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourname/unet-medskin.git
   cd unet-medskin
