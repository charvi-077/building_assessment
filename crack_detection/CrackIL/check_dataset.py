import os
import random
import shutil

train_images = os.listdir("/scratch/kushagra0301/CrackDataset1/train/images")
train_labels = os.listdir("/scratch/kushagra0301/CrackDataset1/train/labels")

random.shuffle(train_images)
# os.makedirs('/scratch/kushagra0301/CrackDataset40')


mod_images = train_images[:int(0.3*len(train_images))]

for i in mod_images:

    shutil.copy("/scratch/kushagra0301/CrackDataset1/train/images/"+i, "/scratch/kushagra0301/CrackDataset30/train/images/"+i)
    shutil.copy("/scratch/kushagra0301/CrackDataset1/train/labels/"+i.replace(".jpg", ".png"), "/scratch/kushagra0301/CrackDataset30/train/labels/"+i.replace(".jpg", ".png"))





# for i in val_images:
#     if i.replace(".jpg", ".png") not in val_labels:
#         os.remove("/scratch/kushagra0301/CrackDataset1/val/images/" + i)

# for i in val_labels:
#     if i.replace(".png", ".jpg") not in val_images:
#         os.remove("/scratch/kushagra0301/CrackDataset1/val/labels/" + i)