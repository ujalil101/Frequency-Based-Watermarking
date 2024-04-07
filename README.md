## Title: A Frequency-Based Non-Intrusive Watermarking Method

### Abstract: 
Watermarking is key for protecting against art forgery and ensuring photo copyrights, but it often changes the original image so much that it can lower its quality. In this discussion, we're introducing a method of watermarking that works in the frequency domain, providing a less intrusive way of adding protection. This technique aims to safeguard the security and authenticity of digital media without sacrificing the image's original quality.

## Python Modules

### watermarking_tools.py:
This module contains functions for adding and extracting text watermarks from images, as well as calculating the Structural Similarity Index (SSIM) between images.

```
    add_text_watermark(original_img_path, text):
```
Description: Adds a text watermark to an image in the frequency domain.
Parameters:
- original_img_path: Path to the original image.
- text: Text to be used as the watermark.
What the function returns:
- original_image: The original grayscale image.
- original_fft: FFT (Fast Fourier Transform) of the original image.
- text_watermarked_image: Image with the text watermark applied.
- text_watermarked_fft: FFT of the watermarked image.

```
extract_text_from_watermarked_image(watermarked_image, original_fft):
```
Description: Extracts the text watermark from a watermarked image.
Parameters:
- watermarked_image: Image with the text watermark.
- original_fft: FFT of the original image.
What the function returns:
- extracted_text_image: Image containing the extracted text regions.

```
calculate_ssim(original_image, watermarked_image):
```
Description: Calculates the Structural Similarity Index (SSIM) between two images.
Parameters:
- original_image: The original grayscale image.
- watermarked_image: Image with the text watermark.
What the function returns:
- ssim: The SSIM value indicating the similarity between the two images.

### visualize.py

```
plot_images_and_3d_fft(original_image, original_fft, text_watermarked_image, text_watermarked_fft):
```
Description: This function plots the original image, its Fourier Transform (FT), the watermarked image, and 3D representations of their FTs.
Parameters:
- original_image: The original grayscale image.
- original_fft: FFT (Fast Fourier Transform) of the original image.
- text_watermarked_image: Image with the text watermark applied.
- text_watermarked_fft: FFT of the watermarked image.

## Presentation Demo
https://docs.google.com/presentation/d/1FGfVa4J5FgCU1dKTGjOTltBaQ4gzcFvVQs_boJee2UY/edit?usp=sharing

