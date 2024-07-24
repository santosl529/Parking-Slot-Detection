from ultralytics import YOLO
import torch
import cv2
import os
import time
import numpy as np
import glob
model = YOLO('runs/detect/train/weights/best.pt')
results = model("videos", conf=0.7)
print(results)

