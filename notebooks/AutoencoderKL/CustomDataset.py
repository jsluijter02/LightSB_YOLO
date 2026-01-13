from torch.utils.data import Dataset
from PIL import Image

# Takes the runtime from 3hrs to 25mins
class customDataset(Dataset):
    def __init__(self, db, transform):
        self.db = db
        self.transform = transform

    def __len__(self):
        return len(self.db)

    def __getitem__(self, i):
        path = self.db[i]['image']
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img
        