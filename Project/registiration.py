import cv2

def image_registration(image1_path, image2_path):
    # Read images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Initialize a brute-force matcher
    bf = cv2.BFMatcher()

    # Match descriptors using KNN (k-nearest neighbors)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to obtain good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image
    cv2.imshow("Image Registration", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_registration("clean.jpg", "susp.jpg")
