# Computer Vision Group Assignment
# README

The project aims to develop an efficient model that uses artificial intelligence and computer vision to distinguish crops from weeds. And establish a user-friendly interface that allows users to easily upload images and obtain recognition results, and supports direct camera input for real-time identification and localization of weeds without affecting crops. At the same time, if more advanced technology is applied, an accurate robot assisted weed control solution can be constructed, which is of great significance for the development of autonomous weed management systems and their application in real life.

We captured multiple growth stages of crops and weeds using an annotated image dataset for training, evaluating, and optimizing the YOLOv3 model to achieve effective classification and detection. And based on this model, a simple and easy-to-use GUI was designed to connect the front-end and back-end, allowing users to upload images and use computer vision technology for real-time image processing, identifying weeds and crop species.
﻿
## Requirements

- Python 3.x
- OpenCV
- Darknet
- YOLO
﻿
## Model Training

 1. setting up GPU:
```python
# clone darknet repo
!git clone https://github.com/AlexeyAB/darknet

# change makefile to have GPU and OPENCV enabled
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile

# verify CUDA
!/usr/local/cuda/bin/nvcc --version

# make darknet (build)
!make

# define helper functions
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

# use this to upload files
def upload():
  from google.colab import files
  uploaded = files.upload()
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)

# use this to download a file
def download(path):
  from google.colab import files
  files.download(path)
```

 2. mounting google drive for data:
```python
# mounting google drive
%cd ..
from google.colab import drive
drive.mount('/content/gdrive')

# this creates a symbolic link so that now the path /content/gdrive/My\ Drive/ is equal to /mydrive
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive
```
From github copy all file from crop_weed_detection_training folder and make Agriculture (same name requires) folder and paste.
```python
!ls /mydrive/Agriculture
```

 3. Downloading dataset:
Please follow the steps below to download and use kaggle data within Google Colab:

(1) Go to you kaggle > account, Scroll to API section and Click Expire API Token(if you have created previously) to remove previous tokens

(2) Click on Create New API Token - It will download kaggle.json file on your machine.

(3) Now just run bellow cell.
```python
!pip install -q kaggle

from google.colab import files

#upload kaggle.json file which you downloaded earlier
files.upload()

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets list

!kaggle datasets download -d ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes

%cd darknet

# unzip the zip file and its contents should now be in /darknet/data/obj
!unzip ../crop-and-weed-detection-data-with-bounding-boxes.zip -d data/

# upload the custom .cfg back to cloud VM from Google Drive
!cp /mydrive/Agriculture/crop_weed.cfg ./cfg

# upload the custom .cfg back to cloud VM from local machine (uncomment to use)
#%cd cfg
#upload()
#%cd ..

# upload the obj.names and obj.data files to cloud VM from Google Drive
!cp /mydrive/Agriculture/obj.names ./data
!cp /mydrive/Agriculture/obj.data  ./data

# upload the obj.names and obj.data files to cloud VM from local machine (uncomment to use)
#%cd data
#upload()
#%cd ..

# upload the generate_train.py script to cloud VM from Google Drive
!cp /mydrive/Agriculture/generate_train.py ./

# upload the generate_train.py script to cloud VM from local machine (uncomment to use)
#upload()

!python generate_train.py

# train.txt file should have to here.
!ls data/agri_data

# upload pretrained convolutional layer weights
!wget http://pjreddie.com/media/files/darknet53.conv.74

# press ctrl+shift+i than paste below code
# open console and paste below code else your runtime will crash after some time
'''
function ClickConnect(){
console.log("Working");
document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
'''
```

 4. Training:
```python
# train your custom detector
!./darknet detector train data/obj.data cfg/crop_weed.cfg darknet53.conv.74 -dont_show

imShow('chart.png')
```

 5. Testing:
```python
# need to set our custom cfg to test mode
%cd cfg
!sed -i 's/batch=32/batch=1/' crop_weed.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' crop_weed.cfg
%cd ..

!ls /mydrive/Agriculture/test

# run your custom detector with this command (upload an image to your google drive to test, thresh flag sets accuracy that detection must be in order to show it)
!./darknet detector test data/obj.data cfg/crop_weed.cfg /mydrive/Agriculture/backup/yolov3_custom_final.weights /mydrive/Agriculture/test/weed_1.jpeg  -thresh 0.3
imShow('predictions.jpg')
```

## GUI System
```python
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter import ttk

# global variable for storing model and image
net = None
LABELS = []
COLORS = []
current_image = None  # for storing the after detecting image
original_image = None  # for storing the origin image
video_stream = None  # for storing the video stream
camera_running = False  # assign the status of camera

# load the model in advanced
def load_yolo_model():
    global net, LABELS, COLORS
    labelsPath = '../data/names/obj.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = '../data/weights/crop_weed_detection.weights'
    configPath = '../data/cfg/crop_weed.cfg'
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# define the function of uploading image
def upload_image():
    global original_image, camera_running  # state the global variable
    if camera_running:
        return
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")])
    if file_path:
        original_image = cv2.imread(file_path)  # save origin image
        detect = original_image.copy()
        detect_image(detect)  # handle copy file
        load_image(detect)

# define the function of loading image
def load_image(img_cv):
    global current_image
    current_image = img_cv.copy()  # save the after detecting image

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((800, 600))
    img_tk = ImageTk.PhotoImage(img_pil)

    clear_display()  # clear the content of screen

    label = tk.Label(frame_display, image=img_tk)
    label.image = img_tk
    label.pack()

    save_button = ttk.Button(frame_display, text="Save Original Image", command=save_image)
    save_button.pack(pady=10, ipadx=10, ipady=5)

    save_result_button = ttk.Button(frame_display, text="Save Result Image", command=save_result_image)
    save_result_button.pack(pady=10, ipadx=10, ipady=5)

# define the function of detecting the image
def detect_image(image):
    global net, LABELS, COLORS
    (H, W) = image.shape[:2]
    confi = 0.5
    thresh = 0.5

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confi:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# define the function of saving the original image
def save_image():
    global original_image
    if original_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                   filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            cv2.imwrite(file_path, original_image)  # save original image
            print(f"the original image has stored to: {file_path}")

# define the function of saving the result image
def save_result_image():
    global current_image
    if current_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                   filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            cv2.imwrite(file_path, current_image)  # save the result image
            print(f"the result image has stored to: {file_path}")

# define the function of changing the statue of camera
def start_stop_camera():
    global video_stream, camera_running
    clear_display()
    if not camera_running:
        video_stream = cv2.VideoCapture(0)  # open camera
        update_frame()
        start_button.config(text="Turn Off Camera")  # update button text
    else:
        stop_camera()
        start_button.config(text="Turn On Camera")  # update button text
    camera_running = not camera_running  # change the status of camera

# define the function of reversing the frame
def update_frame():
    global video_stream
    if video_stream is None:
        return
    ret, frame = video_stream.read()
    if ret:
        frame = cv2.flip(frame, 1)  # reverse horizontally frame
        detect_image(frame)
        load_video_frame(frame)
    root.after(10, update_frame)

# define the function of loading video frame
def load_video_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((800, 600))
    img_tk = ImageTk.PhotoImage(img_pil)

    clear_display()  # clear the content of screen

    label = tk.Label(frame_display, image=img_tk)
    label.image = img_tk
    label.pack()

# define the function of stopping the camera
def stop_camera():
    global video_stream
    if video_stream is not None:
        video_stream.release()
        video_stream = None
    clear_display()  # clear the content

# define the function of clearing display
def clear_display():
    for widget in frame_display.winfo_children():
        widget.destroy()  # clear the content of screen

# define the function of closing the app
def close_app():
    stop_camera()
    root.destroy()  # close the window

# create the window
root = tk.Tk()
root.title("Weed and Crop Detection System")
root.geometry("1200x900")

# load YOLO model
load_yolo_model()

# create the button with improved styles
upload_button = ttk.Button(root, text="Upload Image", command=upload_image, style="TButton")
upload_button.pack(pady=20, ipadx=15, ipady=10)

start_button = ttk.Button(root, text="Turn On Camera", command=start_stop_camera, style="TButton")
start_button.pack(pady=20, ipadx=15, ipady=10)

# create the frame for showing the image
frame_display = tk.Frame(root)
frame_display.pack(pady=10)

# run the cycle
root.protocol("WM_DELETE_WINDOW", close_app)  # release the camera and close the application when closing the window
root.mainloop()
```

## Results

 ![image](https://github.com/user-attachments/assets/c9e49342-60f7-491e-8e3e-c0fa03057dfc)
 
 ![image](https://github.com/user-attachments/assets/356c6581-9806-4b85-8622-f215edad6ed3)
 
 ![image](https://github.com/user-attachments/assets/89f3808c-2c36-43b2-8d99-18ac557182de)
 
 ![image](https://github.com/user-attachments/assets/c4fd9f26-12b1-482d-8ccf-9537e74278e5)
 
 ![image](https://github.com/user-attachments/assets/9f5a6e62-960a-4dba-9990-33860472a64e)
 
 ![image](https://github.com/user-attachments/assets/cd61bbba-0d99-46bc-80d2-ac25693bb9ba)
 
 ![image](https://github.com/user-attachments/assets/d4454d4c-436d-47b4-8bd1-3d7a38eb5a56)
 
## Reference

https://github.com/ravirajsinh45/Crop_and_weed_detection
