## PCA 4096 -> 50 (sklearn recommendation) -> TSNE 50 -> 2
import os,sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scripts.utils import dirs
import numpy as np
import matplotlib.pyplot as plt

def pca_tsne(latents_set:list): # day night lightsb
    labels = np.concatenate([np.zeros(len(latents_set[0])), np.ones(len(latents_set[1])), np.full(len(latents_set[2]), 2)])
    latents = np.vstack(latents_set)

    pca = PCA(n_components=50, random_state=0)
    tsne = TSNE(n_components=2, random_state=0)

    pca_latents = pca.fit_transform(latents)
    tsne_latents = tsne.fit_transform(pca_latents)

    print(tsne_latents.shape)
    return tsne_latents, labels

def plot_tsne(tsne_latents, labels, save_path):
    plt.figure(figsize=(8,6))
    # https://stackoverflow.com/questions/52297857/t-sne-scatter-plot-with-legend
    scatter = plt.scatter(tsne_latents[:,0], tsne_latents[:,1], c = labels)
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, ["Day", "Night", "LightSB Night to Day"])
    plt.title("t-SNE Visualization of Latent BDD100K Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(save_path)
    plt.close()