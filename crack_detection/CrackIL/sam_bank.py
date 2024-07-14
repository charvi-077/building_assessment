import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import json
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import utils
# from torchvision.datasets.folder import default_loader
import numpy as np
# import cv2
import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import time
# import tqdm
from torch import Tensor,nn
from bdd100k import BDD
# import pickle
from sam import SAM
# from sklearn.cluster import KMeans

# bdd_path = "/home/karthik/Downloads/bdd100k"
# print(bdd_path)
# # root_img_path = os.path.join(bdd_path,  "images", "100k")
# root_anno_path = os.path.join(bdd_path, "labels")


# val_img_path = root_img_path + "/val/"
absolutepath = "/scratch/proj"
train_img_path = os.path.join(absolutepath,"bdd100k/images/100k/train")
train_anno_json_path = os.path.join(absolutepath,"bdd100k/labels/det_train.json")
print("Loading files")



dataset_train = BDD(
    train_img_path, train_anno_json_path
)
# if os.path.exists(os.path.join(absolutepath,"memory_bank.pth")):
#     memory_bank = torch.load(os.path.join(absolutepath,"memory_bank.pth"))
memory_bank = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}

sam = SAM()
sam.set_device()
images_done = []
# with open(os.path.join(absolutepath,"images_done.txt"), "r") as fp:
#     images_done = fp.read().splitlines()
# images_done.pop()
image_counter = 1
for idx in range(len(dataset_train)):
    img , target = dataset_train[idx]

    
    if target['image_id'] in images_done:
        continue
    else:
        if target['clear'] == 1:
            for label, box in zip(target['labels'], target['boxes']):
                label = int(label)
                bbox_list = box
                xmin, ymin, xmax, ymax = [int(a) for a in bbox_list]
                mask = np.zeros_like(img)
                mask[:, ymin:ymax, xmin:xmax] = img[:, ymin:ymax, xmin:xmax]
                feature_embedding = sam.get_image_embedding(mask)
                feature_embedding = feature_embedding.detach().cpu().numpy()
                memory_bank[label].append(feature_embedding)
            images_done.append(target['image_id'])
        # print("Done with image: ", target['image_id'])
            with open(os.path.join(absolutepath,"images_done_new_bank.txt"), "w") as fp:
                fp.write("\n".join(images_done))
            if image_counter % 200 == 0:
                torch.save(memory_bank, os.path.join(absolutepath,"memory_bank_new_{}.pth".format(image_counter)))
                image_counter += 1
                memory_bank = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    torch.cuda.empty_cache()
