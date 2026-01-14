import os, sys

def get_base_dir():
    BASE_DIR = os.getcwd()
    while os.path.basename(BASE_DIR) != "LIGHTSB_YOLO":
        BASE_DIR = os.path.dirname(BASE_DIR)
    return BASE_DIR

def is_colab():
    return 'google.colab' in sys.modules

if __name__ == "__main__":
    print(get_base_dir())