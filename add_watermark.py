import cv2
import numpy as np
import random

def add_text_watermark(original_img_path, text):
    # load image
    original_image = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    # convert the text to a binary image 
    # this is to ensure watermarking can be embedded to the FT 
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 1
    text_thickness = 2
    text_color = (255, 255, 255)

    # Generate random coordinates for text position
    max_x = original_image.shape[1] - 1
    max_y = original_image.shape[0] - 1
    text_x = random.randint(0, max_x)
    text_y = random.randint(0, max_y)
    text_position = (text_x, text_y)
    
    # Create text image
    text_image = np.zeros_like(original_image)
    text_width, _ = cv2.getTextSize(text, font, text_size, text_thickness)[0]
    text_position = (text_x, text_y)
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


def extract_text_from_watermarked_image(watermarked_image, original_fft):
    # apply ft
    watermarked_fft = np.fft.fft2(watermarked_image)
    
    # calcualt the differences between watermark and original FT
    diff_fft = watermarked_fft - original_fft
    
    # apply IFT to get the difference image
    diff_image = np.fft.ifft2(diff_fft)
    diff_image = np.abs(diff_image)
    
    # apply thresholding to isolate significant changes
    _, thresholded_diff = cv2.threshold(diff_image, 10, 255, cv2.THRESH_BINARY)
    thresholded_diff = thresholded_diff.astype(np.uint8)
    
    # dilate the thresholded image to fill gaps and enhance text regions
    kernel = np.ones((5, 5), np.uint8)
    thresholded_diff = cv2.dilate(thresholded_diff, kernel, iterations=1)
    
    # find any contours in thresholded image
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # get text regions from contours
    extracted_text_image = np.zeros_like(watermarked_image)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        extracted_text_image[y:y+h, x:x+w] = watermarked_image[y:y+h, x:x+w]  # Extract text regions
    
    return extracted_text_image