import os, sys

def get_base_dir():
    BASE_DIR = os.getcwd()
    while os.path.basename(BASE_DIR) != "LightSB_YOLO":
        print(BASE_DIR)
        BASE_DIR = os.path.dirname(BASE_DIR)
    return BASE_DIR

def get_data_dir():
    base = get_base_dir()
    DATA_DIR = os.path.join(base, "data")
    return DATA_DIR

def get_bdd_dir():
    data = get_data_dir()
    return os.path.join(data, "bdd")

def get_YOLOPX_weights():
    data = get_data_dir()
    return os.path.join(data, "weights", "epoch-195.pth")

def is_colab():
    return 'google.colab' in sys.modules

def add_YOLOPX_to_PATH():
    base = get_base_dir()
    YOLOPX_DIR = os.path.join(base, "models", "YOLOPX")
    sys.path.append(YOLOPX_DIR)

def add_LIGHTSB_to_PATH():
    base = get_base_dir()
    LIGHTSB_DIR = os.path.join(base, "models", "LightSB")
    sys.path.append(LIGHTSB_DIR)

if __name__ == "__main__":
    print(get_base_dir())