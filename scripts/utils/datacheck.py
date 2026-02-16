import os
from PIL import Image

def file_id(file):
    return os.path.basename(file)[0]
# check whether the dirs lign up
def check_directories(dir1, dir2):
    dir1_list = set(file_id(f) for f in os.listdir(dir1))
    dir2_list = set(file_id(f) for f in os.listdir(dir2))
    print("Difference: ", dir1_list-dir2_list)
    return dir1_list - dir2_list 

def check_image_sizes(image_dir, preferred_size=(1280, 720)):
    image_list = os.listdir(image_dir)
    wrong = []
    for img in image_list:
        image = Image.open(os.path.join(image_dir, img))
        if image.size != preferred_size:
            wrong.append[image]
    print(wrong)

    

def check_empty_labels(label_dir):
    pass