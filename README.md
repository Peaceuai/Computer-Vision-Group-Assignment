# Computer Vision Group Assignment
# README

The project aims to develop an efficient model that uses artificial intelligence and computer vision to distinguish crops from weeds. And establish a user-friendly interface that allows users to easily upload images and obtain recognition results, and supports direct camera input for real-time identification and localization of weeds without affecting crops. At the same time, if more advanced technology is applied, an accurate robot assisted weed control solution can be constructed, which is of great significance for the development of autonomous weed management systems and their application in real life.

We captured multiple growth stages of crops and weeds using an annotated image dataset for training, evaluating, and optimizing the YOLOv3 model to achieve effective classification and detection. And based on this model, a simple and easy-to-use GUI was designed to connect the front-end and back-end, allowing users to upload images and use computer vision technology for real-time image processing, identifying weeds and crop species.
﻿
## Requirements

- Python 3.x
- OpenCV
- Pytesseract
- NumPy
﻿
## Installation

 1. Clone the repository:
```sh
git clone https://github.com/PaperCrane28/ComputerVision cd ImageTextExtraction Files
```

2. Install the required packages:
```sh
pip install opencv-python pytesseract numpy
```
![1](https://github.com/user-attachments/assets/aeddd4f1-4944-4da9-aff8-5d10aaa61c15)

﻿
3. Install Tesseract OCR:
- **Windows**: Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).

## Usage
﻿
### Preprocessing and OCR
﻿
The code includes various image preprocessing functions to improve OCR accuracy:
﻿
- Grayscale conversion
- Thresholding
- Erosion
- Deskewing
- Canny edge detection

Sample script to preprocess the image and extract text:
```python
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
﻿
img = cv2.imread('1.jpg')
﻿
# Preprocessing functions

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
﻿
image = cv2.imread('1.jpg')
﻿
gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
deskew = deskew(gray)
canny = canny(gray)
erode = erode(gray)
﻿
# results of OCR with preprocessing

img = gray
cv2.imshow('gray', img)
# Adding custom options
custom_config = r'--oem 1 --psm 6'
pytesseract.image_to_string(img, config=custom_config)

img = thresh
cv2.imshow('thresh', img)
# Adding custom options
custom_config = r'--oem 1 --psm 6'
pytesseract.image_to_string(img, config=custom_config)

img = opening
cv2.imshow('opening', img)
# Adding custom options
custom_config = r'--oem 1 --psm 6'
pytesseract.image_to_string(img, config=custom_config)

img = canny
cv2.imshow('canny', img)
# Adding custom options
custom_config = r'--oem 1 --psm 6'
pytesseract.image_to_string(img, config=custom_config)

img = deskew
cv2.imshow('deskew', img)
# Adding custom options
custom_config = r'--oem 1 --psm 6'
pytesseract.image_to_string(img, config=custom_config)

img = erode
cv2.imshow('erode', img)
# Adding custom options
custom_config = r'--oem 1 --psm 6'
pytesseract.image_to_string(img, config=custom_config)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
﻿
### Recognize Text and Add Boxes
﻿
The script also includes functionality to draw bounding boxes around recognized text:
﻿
```python
d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())
﻿
h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 1)
﻿
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 50:
       (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
       img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('boxes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
﻿
### Reading Text from Image and Saving to File
﻿
This script reads text from an image and outputs the found text to a text file:
﻿
```python
import cv2
import pytesseract

def read_text_from_image(image):
  """Reads text from an image file and outputs found text to text file"""
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Perform OTSU Threshold
  ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
  dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
  contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  image_copy = image.copy()

  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image_copy[y : y + h, x : x + w]
    file = open("results.txt", "a")
    text = pytesseract.image_to_string(cropped)
    file.write(text)
    file.write("\n")
  file.close()
﻿
image = cv2.imread("1.jpg")
read_text_from_image(image)

# OCR results
cv2.imshow('image', image)
f = open("results.txt", "r")
lines = f.readlines()
lines.reverse()
for line in lines:
    print(line)
f.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Results

### Example
 
 Original Image:
 
 ![1](https://github.com/user-attachments/assets/6dae8c17-4505-4a02-a9b2-8b19a675204e)
 
 Display window after image processing：
 
 ![gray](https://github.com/user-attachments/assets/d122146e-1175-4a12-83e1-e43c21263dd4)
![thresh](https://github.com/user-attachments/assets/e9b1698a-ab78-4f97-83f2-5ca5c885ae48)
![opening](https://github.com/user-attachments/assets/39a42bd3-e9d4-4cac-8077-10a2fa692a3e)
![deskew](https://github.com/user-attachments/assets/d0ed33c3-5ce5-4615-9d17-79d86c70ebf0)
![canny](https://github.com/user-attachments/assets/ae2d167c-3944-4120-a5eb-dee9adcb9992)
![erode](https://github.com/user-attachments/assets/66d04839-0026-4156-93ab-ab2db3d3f64a)
![box](https://github.com/user-attachments/assets/5545766b-2dfa-4e4b-bffd-2807b5e3d643)
![boxes](https://github.com/user-attachments/assets/53a9ba89-d364-4ad1-99dc-bdec33ff3ae8)

The extracted text is saved in `results.txt`. You can open and read the file to view the text extracted from the image.
﻿
## Reference

https://colab.research.google.com/github/r3gm/InsightSolver-Colab/blob/main/OCR_with_Pytesseract_and_OpenCV.ipynb#scrollTo=7I0wf49NTSoW
