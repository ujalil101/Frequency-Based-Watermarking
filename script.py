from add_watermark import *
from visualize import *
from PIL import Image




if __name__ == '__main__':
    # path and text
    input_image_path = '/Users/ussie/Desktop/MOCOVI_Research/Research Project/image/mona.jpeg'
    watermark_text = "DR Nemo Rocks"

    # results
    original_image, original_fft, watermarked_image, watermarked_fft = add_text_watermark(input_image_path, watermark_text)
    plot_images_and_3d_fft(original_image, original_fft, watermarked_image, watermarked_fft)
    