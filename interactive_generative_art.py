import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Step 1: Define the Generator Model for DCGAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Step 2: Initialize Generator and Random Noise
generator = Generator()
generator.eval()  # Set generator to evaluation mode

def generate_art(generator, noise_dim=100):
    """Generate an abstract art image from random noise."""
    noise = torch.randn(1, noise_dim, 1, 1)  # Random noise as input
    with torch.no_grad():
        generated_image = generator(noise).detach().cpu().numpy()
    # Scale the output to the 0-255 range for display
    return (generated_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)

# Step 3: Generate and Display Abstract Art
if __name__ == "__main__":
    # Generate an abstract art image
    art_image = generate_art(generator)

    # Display the generated art using Matplotlib
    plt.figure()
    plt.title("Generated Abstract Art")
    plt.axis("off")
    plt.imshow(art_image)
    plt.show()

    # Save the generated image to file
    plt.imsave("generated_abstract_art.png", art_image)
