import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_similarity(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate Structural Similarity Index (SSI)
    ssim_index, _ = ssim(gray1, gray2, full=True)
    
    return ssim_index

def main():
    # Load the first two images
    image1 = cv2.imread('clean.jpg')
    image2 = cv2.imread('clean2.jpg')

    # Check if images are loaded successfully
    if image1 is None or image2 is None:
        print("Error: Could not load images.")
        return

    while True:
        # Load the third image
        current_image = cv2.imread('sus.jpg')

        # Check if the current image is loaded successfully
        if current_image is None:
            print("Error: Could not load the current image.")
            return

        # Calculate similarity between the first two images and the current image
        similarity_percentage = calculate_similarity(image1, image2)

        # Print the similarity percentage
        print(f"Similarity Percentage: {similarity_percentage * 100:.2f}%")

        # Update images for the next iteration
        image1 = image2
        image2 = current_image

if __name__ == "__main__":
    main()
