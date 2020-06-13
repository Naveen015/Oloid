import os
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst

model = FbDeepFace.loadModel()
#model = Facenet.loadModel()

dataset_path = "/home/naveen/datasets/friends/raw"
img_path = "/home/naveen/opencv-face-recognition/images/ross.jpeg"
img = cv2.imread(img_path)
cv2.imshow('find',img)

df = DeepFace.find(img_path = img_path, db_path =dataset_path, model_name = "DeepFace", model = model, enforce_detection=False)

print(df.iloc[0].identity)
img_matched = cv2.imread(df.iloc[0].identity)
cv2.imshow('found',img_matched)
cv2.waitKey(0)
