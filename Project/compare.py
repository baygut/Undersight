# GEBZE TECHNICAL UNIVERSITY GRADUATION PROJECT
# AUTHOR: MUSTAFA BERKAY BAYGUT
# SUPERVISOR: PROF. DR. HABIL KALKAN
# YEAR 2024


import cv2
import numpy as np
from find_best_match import *
import time

def image_registration(img1, img2):
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)


    matches = sorted(matches, key=lambda x: x.distance)

    N = 50
    matches = matches[:N]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # ransac for homography
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    registered_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    return registered_img

def resize_image(image, target_size):
    return cv2.resize(image, target_size)




if __name__ == "__main__":

    start_time = time.time()
    
    #suspicious_image_path = "no_match.jpg" # no match
    #suspicious_image_path = "susp.jpg" # best case
    #suspicious_image_path = "susp2.png" #average
    #suspicious_image_path = "img_5.png" # average 2
    #suspicious_image_path = "sus.jpg" #worst 1
    suspicious_image_path = "sus2.jpg" #worst 2
    #suspicious_image_path = "img3.png" #worst 3
    #suspicious_image_path = "img4.png" #worst 4
    best_match_image, similarity = find_best_match(suspicious_image_path, "database" )
    print(similarity)
    
    print(suspicious_image_path)


    img1 = cv2.imread(best_match_image)
    img2 = cv2.imread(suspicious_image_path)

    registered_image = image_registration(img1, img2)

    target_size = (min(img1.shape[1], img2.shape[1], registered_image.shape[1]), min(img1.shape[0], img2.shape[0], registered_image.shape[0]))
    img1_resized = resize_image(img1, target_size)
    img2_resized = resize_image(img2, target_size)
    registered_image_resized = resize_image(registered_image, target_size)


    diff_image = cv2.absdiff(registered_image_resized, img2_resized)

    gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    _, threshold_diff = cv2.threshold(gray_diff, 70, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = np.zeros_like(diff_image)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if (w > 40 or h > 40) and (w < 200 or h < 200):
            cv2.rectangle(diff_image, (x, y), (x + w, y + h), (0, 0, 255), 8)

    result_image = cv2.addWeighted(diff_image, 1, contour_image, 1, 0)

    if similarity >= 0.2:
        all_images = np.vstack([img1_resized, img2_resized, registered_image_resized, result_image])
    else:
        canvas = np.zeros_like(img1_resized)
        text = "No exact match"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
        text_position = ((canvas.shape[1] - text_size[0]) // 2, (canvas.shape[0] + text_size[1]) // 2)
        cv2.putText(canvas, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        all_images = np.vstack([img1_resized, img2_resized, canvas])

    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    # Display all images in a single window
    cv2.imshow("All Images", all_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
