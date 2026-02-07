import os, sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from scripts.utils import dirs
from models.YOLOPX.lib.core.general import plot_one_box, xywh2xyxy, plot_images

def get_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            entries = line.strip().split()
            cls = int(entries[0])
            x = float(entries[1])
            y = float(entries[2])
            width = float(entries[3])
            height = float(entries[4])
            if len(entries) > 5:
                conf = float(entries[5])
            else:
                conf = 1
            labels.append([cls, x, y, width, height, conf])
    return labels

def filter_conf(predictions, threshold=0.5):
    return [pred for pred in predictions if pred[5] >= threshold]

def plot_bounding_boxes(image_path, label_path, conf_threshold=0.5, color=(255,0,0)):
    labels = filter_conf(get_labels(label_path), threshold=conf_threshold)
    print("labels: ", labels)
    image = cv2.imread(image_path)
    img_height, img_width, img_channels = image.shape

    for label in labels:
        box = label[1:5]
        x, y, w, h = box
        print(box)
        box[0] = (x-w/2)*img_width
        box[1] = (y-h/2)*img_height
        box[2] = (x+w/2)*img_width  
        box[3] = (y+h/2)*img_height 
        plot_one_box(box, image, color=color, line_thickness=2)
    
    return image

def plot_prediction_gt_boxes(image_path, pred_label_path, output_path, conf_threshold=0.5):
    pred_image = plot_bounding_boxes(image_path, pred_label_path, output_path, conf_threshold=conf_threshold, color=(0,255,0))

    gt_image_path = os.path.join(dirs.get_data_dir(), "images", os.path.basename(image_path))
    gt_label_path = os.path.join(dirs.get_data_dir(), "labels", os.path.basename(pred_label_path))
    gt_image = plot_bounding_boxes(gt_image_path, gt_label_path, output_path, conf_threshold=conf_threshold, color=(255,0,0))

    return pred_image, gt_image