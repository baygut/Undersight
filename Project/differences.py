import cv2
import numpy as np


def find_differences(image1, image2, output_path):
    # Read the images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # Ensure that the images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # Compute the absolute difference between the images
    diff = cv2.absdiff(img1, img2)

    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to identify regions with differences
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the differing regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Make rectangles larger (add 10 pixels in each direction)
        x -= 10
        y -= 10
        w += 20
        h += 20

        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Save the output image
    cv2.imwrite(output_path, img2)

    # Save the output image
    cv2.imwrite(output_path, img2)

if __name__ == "__main__":
    image1_path = "clean.jpg"  # Replace with the path to your first image
    image2_path = "susp.jpg"  # Replace with the path to your second image
    output_path = "differences.jpg"  # Replace with the desired output path
    find_differences(image1_path, image2_path, output_path)
