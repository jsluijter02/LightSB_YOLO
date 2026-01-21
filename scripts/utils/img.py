import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def sample_indices(length, how_many=5):
    return np.random.choice(length, how_many, replace=False)

def load_imgs(img_dir, filenames):
    out = []
    for file in filenames:
        path = os.path.join(img_dir, file)
        img = Image.open(path).convert("RGB")
        out.append(img)
    return out

def plot_samples(images, title=""):
    length = len(images)

    plt.figure(figsize=(3*length, 2))
    for i, image in enumerate(images):
        plt.subplot(1,length,i+1)
        plt.imshow(image)
        plt.axis('off')
    
    plt.suptitle(title)
    plt.show()