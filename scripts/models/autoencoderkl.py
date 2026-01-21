import os, sys
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import diffusers
from diffusers import AutoencoderKL

import scripts.utils.dirs as dirs
from scripts.utils.device import get_device

class AutoencoderKL_BDD:
    def __init__(self, url="https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"):
        # source: https://huggingface.co/docs/diffusers/api/models/autoencoderkl
        self.model = AutoencoderKL.from_single_file(url)
        self.model = self.model.to(get_device())

        # no training, so set to eval mode
        self.model.eval()

    def get_latents(self, db, split):
        assert(split in ["train_day", "train_night", "val_day", "val_night"]) 

        latents = []

        transf = transforms.Compose(
            [
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # model expects inputs in range [-1, 1], otherwise the image looks white
            ]
        )
        
        dataset = customDataset(db, transf)

        loader = DataLoader(dataset, batch_size=8, num_workers=4, persistent_workers=True)
        
        filenames_array = []
        device = get_device()
        with torch.no_grad():
            for batch, filenames in tqdm(loader):
                batch = batch.to(device)

                enc = self.model.encode(batch)
                enc_latent = enc.latent_dist.sample().detach()

                enc_latent = enc_latent.flatten(start_dim=1)
                enc_latent = enc_latent.cpu().numpy()
                
                latents.append(enc_latent) 
                filenames_array.extend(filenames)

        latents_array = np.concatenate(latents)
        return latents_array, filenames_array

    def save_latents(self, latents_array, filenames_array, encoding_filename):
        encodings_path = os.path.join(dirs.get_data_dir(), "encodings", encoding_filename)
        os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
        np.savez_compressed(encodings_path, latents=latents_array, filenames=np.asarray(filenames_array))

    def decode_latents(self, latents, batch_size):
        latents = torch.as_tensor(latents, device=get_device())
        num_instances = latents.shape[0]
        latents = latents.reshape([num_instances, 4, 32, 32]) # back to autoencoderkl shape

        # batch
        latents_batches = torch.split(latents, batch_size)

        decoded_latents = []
        for latent in latents_batches:
            decoded = self.model.decode(latent)
            decoded = decoded.sample.detach()
            decoded_latents.append(decoded)
        
        decoded_latents = torch.concat(decoded_latents)
        return decoded_latents

    def save_imgs(self, decoded_latents, filenames):
        data = dirs.get_data_dir()
        save_path = os.path.join(data, "LightSB_images")
        os.makedirs(save_path, exist_ok=True)

        for latent, filename in zip(decoded_latents, filenames):
            save_image(latent, filename)


# Takes the runtime from 3hrs to 25mins
class customDataset(Dataset):
    def __init__(self, db, transform):
        self.db = db
        self.transform = transform

    def __len__(self):
        return len(self.db)

    def __getitem__(self, i):
        path = self.db[i]['image']
        filename = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, filename
        