from watermarking_tools import *
from visualize import *
from PIL import Image




if __name__ == '__main__':
    # path and text
    input_image_path = '/Users/ussie/Desktop/MOCOVI_Research/Research Project/image/Starry_Night.jpeg'
    
    watermark_text = "Usman Jalil"

    # results
    original_image, original_fft, watermarked_image, watermarked_fft = add_text_watermark(input_image_path, watermark_text)
    plot_images_and_3d_fft(original_image, original_fft, watermarked_image, watermarked_fft)
    
    # get text from watermarked image
    extracted_text_image = extract_text_from_watermarked_image(watermarked_image, original_fft)

    # display the extracted text image
    cv2.imshow('Extracted Text Image', extracted_text_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ssim = calculate_ssim(original_image, watermarked_image)
    print("Structural Similarity Index (SSI):", ssim)
    print()

