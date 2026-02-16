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

def plot_samples(images, title="", labels=None, square=True, save_dir:str = None):
    length = len(images)

    if labels != None:
        assert(len(images) == len(labels))

    if square:
        size = (3*length, 4)
    else:
        size = (4*length, 3) 

    plt.figure(figsize=size)
    for i, image in enumerate(images):
        ax = plt.subplot(1,length,i+1)
        ax.imshow(image)
        ax.axis('off')
        if labels != None:
            ax.set_title(labels[i]) 
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)
    plt.show()