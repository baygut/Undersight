import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def compare_images(image1, image2):
    image1 = cv2.resize(image1, (300, 300))
    image2 = cv2.resize(image2, (300, 300))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    ssi_index, _ = ssim(gray_image1, gray_image2, full=True)
    mse_value = mse(gray_image1, gray_image2)

    return ssi_index, mse_value

def find_best_match(input_image, image_folder):
    input_image = cv2.imread(input_image)
    input_image = cv2.resize(input_image, (300, 300))
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_similarity = float('-inf')

    for filename in os.listdir(image_folder):
        filepath = os.path.join(image_folder, filename)

        if os.path.isfile(filepath) and filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            compared_image = cv2.imread(filepath)
            ssi_index, _ = compare_images(input_image, compared_image)

            if ssi_index > best_similarity:
                best_similarity = ssi_index
                best_match = filepath

    return best_match, best_similarity



def find_best_match_res():
    input_image_path = "sus2.jpg"
    images_folder_path = "database"

    best_match_image, similarity = find_best_match(input_image_path, images_folder_path)

    print(f"Best match: {best_match_image}")
    print(f"Structural Similarity Index: {similarity}")
