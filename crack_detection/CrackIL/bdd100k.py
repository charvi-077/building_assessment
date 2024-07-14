#BDD100K loader as per the format expected by DETR

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import os
# from utils.visualize import plot_image

class BDD(Dataset):

    def __init__(self, img_path, anno_json_path, transforms=None):
        self.img_path = img_path
        self.anno_json_path = anno_json_path
        self.transforms = transforms
        self.classes = {'pedestrian': 1, 'rider': 2, 'car': 3, 'truck': 4, 'bus': 5, 'train': 6, 'motorcycle': 7, 'bicycle': 8, 'traffic light': 9, 'traffic sign': 10}
        self._get_data()

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = cv2.imread(os.path.join(self.img_path, image_id))
        target = self.targets[image_id]
        return img, target

    def _get_data(self):
        
        self.ids = []
        self.targets = {}
        with open(self.anno_json_path) as f:
            data = json.load(f)
        
        for i in range(len(data)):

            if 'labels' in data[i]:
                clear_flag = 0
                if data[i]['attributes']['timeofday'] == 'daytime':
                    clear_flag = 1
                bboxes = []
                labels = []
                self.ids.append(data[i]['name'])
                for j in range(len(data[i]['labels'])):
                    if 'box2d' in data[i]['labels'][j]:
                        xmin = data[i]['labels'][j]['box2d']['x1']
                        ymin = data[i]['labels'][j]['box2d']['y1']
                        xmax = data[i]['labels'][j]['box2d']['x2']
                        ymax = data[i]['labels'][j]['box2d']['y2']
                        bboxes.append([xmin, ymin, xmax, ymax])
                        category = data[i]['labels'][j]['category']
                        if category not in self.classes:
                            continue
                        cls = self.classes[category]
                        labels.append(cls)
                self.targets[data[i]['name']] = {'boxes': bboxes, 'labels': labels, 'clear': clear_flag, 'image_id': data[i]['name']}
                

    def __len__(self):
        return len(self.ids)
    


if __name__ == "__main__":

    img_path = "./bdd100k/images/100k/train"
    anno_path = "./bdd100k/labels/det_train.json"

    dataset = BDD(img_path, anno_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


    for i, (img, target) in enumerate(dataloader):

        if i<10:
            print(target['clear'], target['image_id'])