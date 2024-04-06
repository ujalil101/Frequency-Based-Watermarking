import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_images_and_3d_fft(original_image, original_fft, text_watermarked_image, text_watermarked_fft):
    # display four plots to show the original image, ft of original image, ft of watermarked iamge, and then watermarked image (IFT)
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(np.log(1 + np.abs(original_fft)), cmap='gray')
    plt.title('Fourier Transform (Original)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(np.log(1 + np.abs(text_watermarked_fft)), cmap='gray')
    plt.title('Fourier Transform (Watermarked)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(text_watermarked_image, cmap='gray')
    plt.title('Watermarked Image')
    plt.axis('off')
    
    plt.show()
    
    # meshgrid for 3D plots for visualzietion 
    rows, cols = original_image.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    X, Y = np.meshgrid(x, y)
    
    # plot for 3D FT of original image
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, np.log(1 + np.abs(original_fft)), cmap='viridis')
    ax.set_title('Fourier Transform (Original)')
    ax.set_xlabel('X Frequency')
    ax.set_ylabel('Y Frequency')
    ax.set_zlabel('Magnitude')
    
    # plot for 3D FT of watermarked image
    text_watermarked_fft_shifted = np.fft.fftshift(text_watermarked_fft)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, np.log(1 + np.abs(text_watermarked_fft_shifted)), cmap='viridis')
    ax.set_title('Fourier Transform (Watermarked)')
    ax.set_xlabel('X Frequency')
    ax.set_ylabel('Y Frequency')
    ax.set_zlabel('Magnitude')
    
    plt.tight_layout()
    plt.show()