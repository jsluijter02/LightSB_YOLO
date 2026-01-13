import os

def get_base_dir():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    assert(os.path.basename(BASE_DIR) == "LightSB_YOLO")
    return BASE_DIR

if __name__ == "__main__":
    print(get_base_dir())