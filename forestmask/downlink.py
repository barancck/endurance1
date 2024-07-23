import cv2
import numpy as np

image = cv2.imread('./TrueColor/2.jpg')

if image is None:
    print("Error loading image. Is the file path correct?")
else:
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(image, image, mask=mask)

    def adjust_brightness_contrast(image, alpha=1.5, beta=30):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    adjusted_result = adjust_brightness_contrast(result, alpha=1.5, beta=30)

    green_area = cv2.countNonZero(mask)
    total_area = mask.size
    green_percentage = (green_area / total_area) * 100

    if green_percentage > 20:
        print("Forest or greenery area detected.")
    else:
        print("Forest or greenery area not detected.")

    cv2.imshow('Mask', mask)
    cv2.imshow('Original Image', image)
    cv2.imshow('Result', result)
    cv2.imshow('Adjusted Result', adjusted_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
