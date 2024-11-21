import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pygame

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

# Step 2: Initialize Generator and Helper Function
generator = Generator()
generator.eval()  # Set generator to evaluation mode

def generate_art(generator, noise_dim=100):
    """Generate an abstract art image from random noise."""
    noise = torch.randn(1, noise_dim, 1, 1)  # Random noise as input
    with torch.no_grad():
        generated_image = generator(noise).detach().cpu().numpy()
    # Scale the output to the 0-255 range for display
    return (generated_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)

# Step 3: Dynamic Visual Performances with Pygame
def dynamic_visual_performance(generator):
    """Display and regenerate dynamic visuals interactively."""
    pygame.init()
    screen_width, screen_height = 800, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Dynamic Visual Performances")
    clock = pygame.time.Clock()

    # Generate the first abstract art image
    art_image = generate_art(generator)
    art_surface = pygame.surfarray.make_surface(art_image)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Regenerate art when spacebar is pressed
                if event.key == pygame.K_SPACE:
                    art_image = generate_art(generator)
                    art_surface = pygame.surfarray.make_surface(art_image)

        # Display the generated art
        screen.blit(art_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)  # Limit to 30 frames per second

    pygame.quit()

# Step 4: Save and Display a Static Image
if __name__ == "__main__":
    # Generate and display a static art image
    static_art = generate_art(generator)
    plt.figure()
    plt.title("Generated Abstract Art")
    plt.axis("off")
    plt.imshow(static_art)
    plt.show()

    # Save the static art image
    plt.imsave("generated_abstract_art.png", static_art)

    # Launch dynamic visual performances
    dynamic_visual_performance(generator)
