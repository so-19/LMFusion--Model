import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import UNetDownsampler, UNetUpsampler

def test_unet_components():
    torch.manual_seed(42)
    batch_size = 2
    input_dim = 4
    hidden_dim = 64
    img_size = 16
    input_image = torch.randn(batch_size, input_dim, img_size, img_size)
    print(f"Input image shape: {input_image.shape}")
    timesteps = torch.rand(batch_size)
    print(f"Timesteps shape: {timesteps.shape}")
    
    downsampler = UNetDownsampler(input_dim, hidden_dim)
    upsampler = UNetUpsampler(hidden_dim, input_dim)
    
    latent = downsampler(input_image, timesteps)
    print(f"Downsampled latent shape: {latent.shape}")
    
    downsampled_size = img_size // 2
    expected_seq_len = downsampled_size * downsampled_size
    expected_latent_shape = (batch_size, expected_seq_len, hidden_dim)
    print(f"Expected latent shape: {expected_latent_shape}")
    assert latent.shape == expected_latent_shape, f"Expected {expected_latent_shape}, got {latent.shape}"
    output = upsampler(latent, timesteps, input_image)
    print(f"Upsampled output shape: {output.shape}")
    
    expected_output_shape = (batch_size, input_dim, img_size, img_size)
    print(f"Expected output shape: {expected_output_shape}")
    assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output.shape}"
    
    visualize_results(input_image, output)
    print("UNet components test passed!")
    return input_image, latent, output

def visualize_results(input_image, output_image):
    sample_idx = 0
    input_np = input_image[sample_idx].detach().cpu().numpy()
    output_np = output_image[sample_idx].detach().cpu().numpy()
    def normalize_for_display(img):
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = img.squeeze(2)
        else:
            if img.shape[2] > 3:
                img = img[:, :, :3]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    input_viz = normalize_for_display(input_np)
    output_viz = normalize_for_display(output_np)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    if len(input_viz.shape) == 2:
        plt.imshow(input_viz, cmap='gray')
    else:
        plt.imshow(input_viz)
    plt.title("Input Image (First 3 channels)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if len(output_viz.shape) == 2:
        plt.imshow(output_viz, cmap='gray')
    else:
        plt.imshow(output_viz)
    plt.title("Output Image (First 3 channels)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("unet_test_results.png")
    print("Visualization saved as 'unet_test_results.png'")

def test_noise_reconstruction():
    torch.manual_seed(42)
    batch_size = 1
    input_dim = 4
    hidden_dim = 64
    img_size = 16
    x = torch.linspace(-1, 1, img_size)
    y = torch.linspace(-1, 1, img_size)
    xx, yy = torch.meshgrid(x, y)
    pattern = torch.sin(xx * 3) * torch.cos(yy * 3)
    input_image = pattern.unsqueeze(0).unsqueeze(0).repeat(batch_size, input_dim, 1, 1)
    
    noise_level = 0.3
    noisy_input = input_image + noise_level * torch.randn_like(input_image)
    timesteps = torch.rand(batch_size)
    downsampler = UNetDownsampler(input_dim, hidden_dim)
    upsampler = UNetUpsampler(hidden_dim, input_dim)

    latent = downsampler(noisy_input, timesteps)
    output = upsampler(latent, timesteps, noisy_input)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    clean_viz = input_image[0, 0].detach().cpu().numpy()
    plt.imshow(clean_viz, cmap='viridis')
    plt.title("Original Clean Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    noisy_viz = noisy_input[0, 0].detach().cpu().numpy()
    plt.imshow(noisy_viz, cmap='viridis')
    plt.title(f"Noisy Input (noise level: {noise_level})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    output_viz = output[0, 0].detach().cpu().numpy()
    plt.imshow(output_viz, cmap='viridis')
    plt.title("Reconstructed Output")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("unet_noise_reconstruction.png")
    print("Noise reconstruction test visualization saved as 'unet_noise_reconstruction.png'")
    mse = torch.mean((output - noisy_input) ** 2).item()
    print(f"Mean Squared Error between input and output: {mse:.6f}")
    
    return noisy_input, latent, output

if __name__ == "__main__":
    print("Testing UNet components...")
    test_unet_components()
    print("\nTesting noise reconstruction...")
    test_noise_reconstruction()
    print("\nAll tests completed!")
