import glob
import os
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from data_utils.augment import Augmentation, mirrorPadding2D
import pandas as pd
from PIL import Image

class WSIDataset(Dataset):
    def __init__(self, root_dir, mode, taskList=None, total_fold=5, valid_fold=2, miniScale=1):
        """
        Dataset class for LEVIR-CD dataset
        
        Args:
            root_dir: Dictionary or string path to dataset root
            mode: 'train', 'val', or 'test'
            taskList: Not used in current implementation
            total_fold: Total number of folds for cross-validation
            valid_fold: Validation fold number
            miniScale: Scale factor for images
        """
        self.root_dir = root_dir
        self.mode = mode
        self.miniScale = miniScale
        self.total_fold = total_fold
        self.valid_fold = valid_fold
        
        # Initialize lists for image paths
        self.all_png_dir_1 = []  # T1 images
        self.all_png_dir_2 = []  # T2 images  
        self.all_label_change = []  # Labels
        
        # Handle both dictionary and string inputs for root_dir
        if isinstance(root_dir, dict):
            # Original code logic for multiple datasets
            for k, v in self.root_dir.items():
                self.all_png_dir_1 += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "T1" + os.sep + '*'))
                self.all_png_dir_2 += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "T2" + os.sep + '*'))
                self.all_label_change += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "label" + os.sep + '*'))
        else:
            # Single dataset path (for Kaggle usage)
            self.all_png_dir_1 = sorted(glob.glob(os.path.join(root_dir, mode, "T1", '*')))
            self.all_png_dir_2 = sorted(glob.glob(os.path.join(root_dir, mode, "T2", '*')))
            self.all_label_change = sorted(glob.glob(os.path.join(root_dir, mode, "label", '*')))
        
        # Extract filenames for matching
        self.all_png_dir_1_name = [os.path.splitext(os.path.split(i)[1])[0] for i in self.all_label_change]
        
        print(f"T1 patch numbers: {len(self.all_png_dir_1)}")
        print(f"T2 patch numbers: {len(self.all_png_dir_2)}")
        print(f"Label patch numbers: {len(self.all_label_change)}")
        
        # Training parameters
        self.isTrain = False
        self.source_size = (256, 256)
        self.randomImgSizeList = [(256, 256)]
        self.randomImgSizeList = self.randomImgSizeList[::1]
        self.randomImgSize = (256, 256)
    
    def __getitem__(self, index):
        """
        Get a single item from the dataset
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            tuple: (img1, img2, label1, label2, labelc, dir)
        """
        dir_name = self.all_png_dir_1_name[index]
        img1_path = self.all_png_dir_1[index]
        img2_path = self.all_png_dir_2[index]
        labelc_path = self.all_label_change[index]
        
        if self.mode == "train":
            # Load and resize images for training
            img1 = np.array(Image.open(img1_path).resize(self.randomImgSize))
            img2 = np.array(Image.open(img2_path).resize(self.randomImgSize))
            labelc = np.expand_dims(np.array(Image.open(labelc_path).resize(self.randomImgSize)), axis=2)
            
            # Apply mirror padding for augmentation
            img1 = mirrorPadding2D(img1)
            img2 = mirrorPadding2D(img2)
            labelc = mirrorPadding2D(labelc)
            
            # Convert back to PIL Images
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)
            labelc = Image.fromarray(np.squeeze(labelc))
            
            # Apply geometric augmentations
            aug = Augmentation()
            img2_combine, bias_y, bias_x = aug.randomSpaceAugment(
                [img1, img2, labelc], 
                source_size=self.randomImgSize, 
                unoverlap=None
            )
            
            # Unpack augmented images
            img1, img2, labelc = img2_combine
            
            # Apply photometric distortions
            imgPhotometricDistortion1 = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            imgPhotometricDistortion2 = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            img1 = imgPhotometricDistortion1(img1)
            img2 = imgPhotometricDistortion2(img2)
            labelc = torch.FloatTensor(np.array(labelc)) / 255
            
        elif self.mode in ["validation", "val", "test"]:
            # Load and process images for validation/test
            img1 = Image.open(img1_path).resize(self.randomImgSize)
            img2 = Image.open(img2_path).resize(self.randomImgSize)
            
            imgTransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            labelc = np.expand_dims(np.array(Image.open(labelc_path).resize(self.randomImgSize)), axis=2)
            labelc = torch.FloatTensor(np.squeeze(labelc)) / 255
            
            img1 = imgTransforms(img1)
            img2 = imgTransforms(img2)
        
        # Dummy labels (not used in current implementation)
        label1 = torch.FloatTensor([0])
        label2 = torch.FloatTensor([0])
        
        return img1, img2, label1, label2, labelc, dir_name
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.all_png_dir_1)

if __name__ == "__main__":
    # Example usage
    dataset_path = "/kaggle/working/processed_data"
    
    # Test dataset creation
    train_dataset = WSIDataset(root_dir=dataset_path, mode="train")
    val_dataset = WSIDataset(root_dir=dataset_path, mode="val") 
    test_dataset = WSIDataset(root_dir=dataset_path, mode="test")
    
    print("Dataset created successfully!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Test loading a single sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample shapes: img1={sample[0].shape}, img2={sample[1].shape}, label={sample[4].shape}")
