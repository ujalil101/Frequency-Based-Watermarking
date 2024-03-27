import cv2
import numpy as np

def add_text_watermark(original_img_path, text):
    # load image
    original_image = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    
    # convert the text to a binary image 
    # this is to ensure watermarking can be embedded to the FT 
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 1
    text_thickness = 2
    text_color = (255, 255, 255)
    text_image = np.zeros_like(original_image)
    text_width, _ = cv2.getTextSize(text, font, text_size, text_thickness)[0]
    text_position = ((text_image.shape[1] - text_width) // 2, text_image.shape[0] // 2)
    cv2.putText(text_image, text, text_position, font, text_size, text_color, text_thickness)
    
    # apply FT
    original_fft = np.fft.fft2(original_image)
    text_fft = np.fft.fft2(text_image)
    
    # add text in frequency domain
    alpha = 0.1  # control the strength of the text
    text_watermarked_fft = original_fft + alpha * text_fft
    
    # apply IFT
    text_watermarked_image = np.fft.ifft2(text_watermarked_fft)
    text_watermarked_image = np.abs(text_watermarked_image)
    
    # normalize result to uint8 (FT values may not be between 0-255) 
    text_watermarked_image = cv2.normalize(text_watermarked_image, None, 0, 255, cv2.NORM_MINMAX)
    text_watermarked_image = text_watermarked_image.astype(np.uint8)
    
    return original_image, original_fft, text_watermarked_image, text_watermarked_fft
