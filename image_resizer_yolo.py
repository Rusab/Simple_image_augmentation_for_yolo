# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:46:27 2021

@author: Rusab
"""


import random
import copy
import cv2
from matplotlib import pyplot as plt
import os
import albumentations as A


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):

    x_center, y_center, w, h = bbox
    
    x_center = width*x_center
    y_center = height*y_center
    w = width*w
    h = height*h
    
    
    x_min, x_max, y_min, y_max = int(x_center - (w//2)), int(x_center + (w//2)), int(y_center - (h//2)), int(y_center + (h//2))

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, class_names):
    img = image.copy()
    for bbox in bboxes:
        class_name = class_names[bbox[0]]
        img = visualize_bbox(img, bbox[1:5], class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)






folder = os.getcwd()
destination = os.path.join(folder, 'Augmented Files')

if not os.path.exists(destination):
    os.makedirs(destination)

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[-3:] == 'jpg']
files = list(set([f[:-4] for f in files]))


errors = []
for img in files:
    img_name = img +'.jpg'
    label_name = img + '.txt'

    image = cv2.imread(os.path.join(folder, img_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(os.path.join(folder, img_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    with open(os.path.join(folder, label_name)) as f:
        lines = f.readlines()
    
    bboxes = []
    for line in lines:
        cord = line.split(" ")
        cord[4] = cord[4][:-1]
        cord[0] = int(cord[0])
        cord[1:5] = [float(c) for c in cord[1:5]]
        bboxes.append(cord)
    

    category_ids = []
    to_transboxes = copy.deepcopy(bboxes)
    for bbox in to_transboxes:
        category_ids.append(bbox.pop(0))
    

    transform = A.Compose(
        [
        A.Resize(416, 416)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])
    )
    random.seed(7)
    num = 0
    
    
    try:
        transformed = transform(image=image, bboxes=to_transboxes, category_ids=category_ids)
    except:
        print("Couldn't agument:", img_name)
        errors.append(img_name)
    
    for tbbox, bbox in zip(transformed['bboxes'], bboxes):
        x = [round(a, 4) for a in tbbox]
        bbox[1:5] = x
    save_name = img + 'aug' + str(num)
    cv2.imwrite(os.path.join(destination, save_name + '.jpg'), transformed['image'])
    label = open(os.path.join(destination, save_name + '.txt'), 'w+')
    for bbox in bboxes:
        for i in range(0, 5):
            if i != 0:
                #print(i)
                label.write(' ')
            label.write(str(bbox[i]))
        label.write("\n")
        
    label.close()
    
log = open(os.path.join(destination, 'log' + '.txt'), 'w+')
log.write("Couldn't augment these files:\n")
for error in errors:
    log.write(error)
    log.write("\n")

log.close()
   
    
    
