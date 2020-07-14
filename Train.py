import argparse
import segmentation_models_pytorch as smp
import torch
import numpy as np
import segmentation_models_pytorch as smp
torch.cuda.set_device(1)
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset
import albumentations as albu
import cv2
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.utils.tensorboard import SummaryWriter
from datetime import date 

class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name
class Metric(BaseObject):
    pass
class meanIoU(Metric):
    __name__ = 'mean_iou'
    
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def forward(self, outputs, labels):
        SMOOTH = 1e-6
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        intersection = ((outputs*255).int() & (labels*255).int()).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = ((outputs*255).int() | (labels*255).int()).float().sum((1, 2))         # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded.mean()


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annotation, isize = (384,512), transform=None, target_transform=None, transforms=None,augmentation=None, 
            preprocessing=None,):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.ids = [0,4,6,7,9,10,11,12,13,17,21,23,27,28]
        self.size = isize
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.size[1],self.size[0]))
        image = img
        #if self.transforms is not None:
            #img, target = self.transforms(img, target)
        mask = np.zeros((11,self.size[0],self.size[1]))
        for m in target:
            mask[m['category_id'],:,:] = np.maximum(mask[m['category_id'],:,:],cv2.resize(coco.annToMask(m)*255,(self.size[1],self.size[0])))
        mask = mask[3:,:,:]
        masks = []
        for i in range(len(mask)):
            masks.append(mask[i])
        if self.augmentation:
            sample = self.augmentation(image=image, masks=masks)
            img = sample['image']
            mask_aug = sample['masks']
            for i in range(len(mask_aug)):
                mask[i,:,:] = mask_aug[i]
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img)
            img = sample['image']
            for i in range(len(mask)):
                sample  = self.preprocessing(image = image, mask=mask[i,:,:])
                mask[i,:,:] = sample['mask']

        return img, mask
    def __len__(self):
        return len(self.ids)

def to_tensor_image(x, **kwargs):
    
    x = x/255.0 
    shape = x.shape
    x = x.reshape((shape[2],shape[0],shape[1]))
    x = x.astype('float32')
    return x

def to_tensor_mask(x, **kwargs):
    x = x/255.0
    x = x.astype('float32')
    return x


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
       albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor_image, mask=to_tensor_mask),
    ]
    return albu.Compose(_transform)

def get_training_augmentation(size1):
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=size1[0], min_width=size1[1], always_apply=True, border_mode=0),
        albu.RandomCrop(height=size1[0], width=size1[1], always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)




def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('root_folder', help='Root folder of dataset')
    parser.add_argument('--model_name', help='Name of classificator')
    parser.add_argument('--encoder_name', help='Name of encoder')
    parser.add_argument('--weight', help='Weights of encoder')
    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--lr', help='Starting learning rate')
    parser.add_argument('--image_size', help='Size for resizing of image')
    parser.add_argument('--batch', help='Batch size for training')
    parser.add_argument('--activation', help='activation')
    parser.add_argument('--nclass', help='Number of classes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_name is not None:
        model_name = args.model_name
    else:
        model_name = "Unet"
    
    if args.encoder_name is not None:
        encoder_name = args.encoder_name
    else:
        encoder_name = "resnet101"


    if args.weight is not None:
        weight = args.weight
    else:
        weight = None
    
    

    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = 100
    
    if args.activation is not None:
        activation = args.activation
    else:
        activation = 'sigmoid'

    if args.nclass is not None:
        nclass = args.nclass
    else:
        nclass = 8
    
    if args.lr is not None:
        lr = args.lr
    else:
        lr = 0.001

    if args.batch is not None:
        batch = args.batch
    else:
        batch = 1
    
    


    print("Model name:",model_name) 
    print("With encoder:",encoder_name)
    print("Training will be perfome on number of epochs:",epochs)
    print("With learning rate:",lr)
    print("Batch size:",batch)
    print("Main activation function:",activation)
    model = getattr(smp, model_name)(
        classes=nclass,
        encoder_name=encoder_name,
        encoder_weights=weight, 
        activation = activation
    )
    DEVICE = torch.device('cuda:1')
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name)
    DATA_DIR = args.root_folder
    x_train_dir = os.path.join(DATA_DIR, 'images')
    y_train_dir = os.path.join(DATA_DIR, 'annotations/instances.json')

    x_valid_dir = os.path.join(DATA_DIR, 'images')
    y_valid_dir = os.path.join(DATA_DIR, 'annotations/instances.json')

    x_test_dir = os.path.join(DATA_DIR, 'images')
    y_test_dir = os.path.join(DATA_DIR, 'annotations/instances.json')

    train_dataset = CocoDetection(root=DATA_DIR,annotation=y_train_dir, isize = (384,512), 
    preprocessing=get_preprocessing(preprocessing_fn))

    valid_dataset = CocoDetection(root=DATA_DIR,annotation=y_train_dir, isize = (384,512),
    preprocessing=get_preprocessing(preprocessing_fn)                              
    )

    train_loader = DataLoader(train_dataset, batch_size= 5, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size= 2, shuffle=False, num_workers=2)
    loss = smp.utils.losses.DiceLoss(activation = activation)
    metrics = [
        smp.utils.metrics.IoU(activation=nn.ReLU(),threshold=0.5),
        smp.utils.metrics.Fscore(activation=nn.ReLU(),threshold=0.5)
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    max_score = 0
    writer = SummaryWriter('runs/' +model_name + "_" + epochs + "_" + encoder_name + '_'+ str(date.today())) 
    for i in range(0, int(epochs)):

        print('\nEpoch: {}'.format(i))
        
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        writer.add_scalar('Dice loss',valid_logs['dice_loss'], i)
        writer.add_scalar('Iou score',valid_logs['iou_score'], i)
        writer.add_scalar('F1 score',valid_logs['fscore'], i)
        #writer.add_scalar('mean Iou score',train_logs['mean_iou'], i) 
        scheduler.step()
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, model_name + "_" + epochs + "_" + encoder_name + '_' + activation +'_'+ str(date.today()) +'.pth')
            print('Model saved!')
if __name__ == '__main__':
    main()