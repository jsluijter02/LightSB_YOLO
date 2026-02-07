## YOLOPX To ultralytics format
import os
import yaml
from tqdm import tqdm

def export_to_ultralytics_format(db, save_dir):
    # only labels, because the images are already in bdd folder / lightsb folder / clahe folder
    os.makedirs(save_dir, exist_ok=True)

    for item in tqdm(db):
        image_path = item['image']
        label = item['label']

        image_filename = os.path.basename(image_path)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(save_dir, label_filename)

        # write lable file
        with open(label_path, 'w') as f:
            for obj in label:
                cls_id = int(obj[0])
                x_center = obj[1]
                y_center = obj[2]
                width = obj[3]
                height = obj[4]
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

def yaml_file(db, train_img_path, val_img_path, yaml_path):
    data = {
        'train': train_img_path,
        'val': val_img_path,
        'nc': 1,
        'names': ["object"]
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)


   

