import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_images(img1, img2, title1="", title2=""):
    fig = plt.figure(figsize=[15,15])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)

path = "./img/1.jpeg"
image = cv2.imread(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 11,90, 90)
edges = cv2.Canny(blur, 30, 100)
cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
_ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
image_reduced_cnts = image.copy()
_ = cv2.drawContours(image_reduced_cnts, cnts, -10, (255,0,255),5)

from tensorflow import keras
model1 = keras.models.load_model('model_akur.h5')

plate = None
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    edges_count = cv2.approxPolyDP(c, 0.05 * perimeter, True)
    if len(edges_count) == 4:
        x,y,w,h = cv2.boundingRect(c)
        plate = image[y:y+h, x:x+w]
        break

cv2.imwrite("plate.png", plate)

import pytesseract
from tensorflow import keras
model2 = keras.models.load_model('model_mnist.h5')
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

text = pytesseract.image_to_string(plate, lang='eng', config='--psm 6')
print("Plat nomor:",text)

import mysql.connector
from mysql.connector import Error
try:
    connection = mysql.connector.connect(host='localhost',
                                         database='sisnoken',
                                         user='root',
                                         password='')
    value = (text)
    query = "INSERT INTO historykeluar (plat) VALUES (%s)"
    cursor = connection.cursor()
    cursor.execute(query,(value,))
    connection.commit()
    print("true")

except mysql.connector.Error as error:
    print("false".format(error))
    
 