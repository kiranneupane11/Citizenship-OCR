import cv2
import pytesseract
import imutils
from pytesseract import Output
import math
from typing import Tuple, Union
import numpy as np
from deskew import determine_skew

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'"C:\Program Files\Tesseract-OCR\tessdata" --psm 7 --oem 3'

def ocr(file):
    """
    This function will handle the core OCR processing of images.
    """
    # Orientation Fix
    file = cv2.imread(file)
    results = pytesseract.image_to_osd(file, config='--psm 0 -c min_characters_to_try=5',output_type=Output.DICT)
    # Rotate the image to correct the orientation
    fixed_orientation = imutils.rotate_bound(file, angle=results["rotate"])

    # Skew Fix
    def rotate(
            image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
    angle = determine_skew(fixed_orientation)
    skewed = rotate(fixed_orientation, angle, (0, 0, 0))

    #Resize Image
    width, height = 1500, 1200
    resized_image = cv2.resize(skewed, (width, height))
    bbox = resized_image.copy()
    faded_image = resized_image.copy()


    # Remove Noise
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 85, 80])
    upper_blue = np.array([158, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    alpha = 0.3 
    neutral_color = (255, 255, 255)
    faded_image[blue_mask > 0] = (
        (1 - alpha) * np.array(neutral_color) + alpha * faded_image[blue_mask > 0]
    ).astype(np.uint8)

    #Feature Extraction
    gray = cv2.cvtColor(faded_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,125,24)    #(165,26) for 4000*3000 size image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    erode = cv2.erode(adaptive_thresh, kernel, iterations=1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (18,5))  #(18,5) standard
    dilate = cv2.dilate(erode,kernel1, iterations=1)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9,5))
    # opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel2, iterations=2)     #(9,5) & 3 iterations removes picture noise better

    # config = "--psm 7 --oem 3"
    lang = "nep"
    result = []

    # Filter contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if aspect_ratio > 1.5 and aspect_ratio < 12 and area > 500 and area < 15000:
            roi = bbox[y:y+h, x:x+w]
            cv2.rectangle(bbox,(x,y),(x+w,y+h),(36, 255, 12), 2)
            text = pytesseract.image_to_string(roi, config=tessdata_dir_config, lang=lang)
            text = [line.strip() for line in text.split("\n") if line.strip() and line != '\x0c']
            for item in text:
                result.append(item)
    return result