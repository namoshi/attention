import random
import numpy as np
import matplotlib.pyplot as plt

import torch

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

    else:
        device = "cpu"

    print("CUDA is available:", torch.cuda.is_available())
    return device


def fixed_r_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_time(interval):
    time = {"time" : "{}h {}m {}s".format(
            int(interval / 3600), 
            int((interval % 3600) / 60), 
            int((interval % 3600) % 60))}
    return time

# show sample 12 imgs
def show_img(save_path, dataloader):
    for batched in dataloader:
        images = batched["image"]
        labels = batched["label"]
        break

    images = (images - images.min()) / (images.max() - images.min())
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(12):
        ax = axes[i // 4, i % 4]
        img = np.transpose(images[i].numpy(), (1, 2, 0))  # (C, H, W) → (H, W, C)
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')

    plt.savefig(save_path)
    plt.close()
