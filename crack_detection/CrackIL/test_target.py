import os
import numpy as np
import torch
from argparse import ArgumentParser
import cv2
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F 
from model.erfnet_RA_parallel import Net as Net_RAP # model
import data.custom_transforms as tr
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CrackTargetNew(Dataset):
    """
    Crack dataset
    """

    def __init__(self, args, base_dir):

        super().__init__()

        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images')

        self.args = args

        self.im_ids = []
        self.images = []

        for image in os.listdir(self._image_dir):
            self.images.append(os.path.join(self._image_dir, image))  # Path of images
            self.im_ids.append(image.split('.')[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        width, height = _img.size
        _img = _img.resize((self.args.crop_size_width, self.args.crop_size_height), Image.BILINEAR)

        sample = {'image': _img}

        sample = self.transform(sample)
        sample['image_id'] = self.im_ids[index]
        sample['image_size'] = (width, height)
        sample['domain'] = 1  # 1 for target
        return sample

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize_new(),
            tr.ToTensor_new()])

        return composed_transforms(sample)

def main(args):

    model = Net_RAP(args.num_classes, args.nb_tasks, args.current_task)
    model = torch.nn.DataParallel(model).cuda()
    print("model keys .....................................")
    print(model.state_dict().keys())
 
    loaded_model = torch.load(args.saved_model)

    model_state_dict = loaded_model['state_dict']
    print("saved model keys ........................................")
    print(model_state_dict.keys())
        
    # Compare the keys in the loaded state dictionary with the keys of the model's state dictionary
    # model_keys = model.state_dict().keys()
    # missing_keys = [key for key in model_keys if key not in model_state_dict]
        
    # if len(missing_keys) == 0:
    #     print("Model architecture matches the saved model.")
    # else:
    #     print("Missing keys in the loaded state dictionary:", missing_keys)

    model.load_state_dict(loaded_model['state_dict'])

    print("Loading weights from: ", args.saved_model)
    target_dataset = CrackTargetNew(args = args, base_dir = args.target_dataset_path)
    print("Target:", len(target_dataset))
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=1)
    eval_model(model, target_loader, args.output_folder, 0)
    
    # for k, v in loaded_model['state_dict'].items():
    #         print(k, v)
    
# -------------- model evaluateion 
def eval_model(model, dataset_loader, output_folder, task):
    model.eval()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with torch.no_grad():
        for step, sample in enumerate(dataset_loader):
            images, image_ids, image_sizes = sample['image'], sample['image_id'], sample['image_size']
            inputs = images.cuda()
            print("Image Sizes:", image_sizes)
            image_sizes_np = np.squeeze(np.array([size.numpy() for size in image_sizes]))
            # print("Image Size 2: ",image_sizes_np ) # it is in tensor
            _, outputs = model(inputs, task)
            pred = outputs.data.max(1)[1].cpu().numpy() 
            original_sizes = [int(size) for size in image_sizes_np]          
            # print(original_sizes) # it is in numpy 
            pred_image = np.transpose(pred, (1, 2, 0))
            pred_image = np.asarray(pred_image*255, dtype=np.uint8)
            pred_image = cv2.resize(pred_image, (original_sizes[0], original_sizes[1]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(output_folder, f'{image_ids}.jpg'), pred_image)
    print('Evaluation done.')

# criterion = CrossEntropyLoss2d()

# average_loss_val_target_current, val_miou_target_current = eval(model, target_loader, criterion, current_task, epoch, evaluator)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--batch_size_val', type=int, default=1, help='input val batch size')
    # parser.add_argument('--source_dataset_path', type=str, default='/scratch/kushagra0301/CrackDataset1', help='source dataset')
    parser.add_argument('--target_dataset_path', type=str, default='/home2/kushagra0301/CrackIL/eval_data/crack/', help='target dataset')
    parser.add_argument('--output_folder', type=str, default='/home2/kushagra0301/CrackIL/eval_data/crack_output/', help='output eval result of only images')
    parser.add_argument('--saved_model', type=str, default='/home2/kushagra0301/CrackIL/model_parameters/model_150.pth.tar', help='saved model')
    parser.add_argument('--num_classes', type=int, default=[2,2], help='number of classes')
    parser.add_argument('--crop_size_height', type=int, default=512, help='crop size height')
    parser.add_argument('--crop_size_width', type=int, default=1024, help='crop size width')
    # parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--nb_tasks', type=int, default=2) 
    parser.add_argument('--current_task', type=int, default=0)
    # parser.add_argument('--save_dir', type=str, default='/scratch/kushagra/crack_il', help='save directory')
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--kld_weight', type=float, default=0.1, help='kld weight')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='ce weight')
    parser.add_argument('--dataset_avoided', type=str, help='dataset avoided')

    args = parser.parse_args()
    main(args)