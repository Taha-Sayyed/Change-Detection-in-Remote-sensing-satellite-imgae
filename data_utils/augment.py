import numpy as np
import random
from PIL import Image
import math
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf

class Augmentation:
    def __init__(self):
        pass
    
    def randomSpaceAugment(self, image, source_size=(256, 256), angle=None, scale=(0.8, 1.25), unoverlap=None):
        """
        Apply random spatial augmentations to a list of images
        
        Args:
            image: List of PIL Images
            source_size: Target size (height, width)
            angle: Rotation angle (if None, random angle is chosen)
            scale: Scale range tuple (min, max)
            unoverlap: Overlap parameter (if None, random cropping is applied)
        """
        if angle is None:
            angle = transforms.RandomRotation.get_params([-180, 180])
        
        if isinstance(angle, list):
            angle = random.choice(angle)
        
        # Rotation
        for i in range(len(image)):
            image[i] = image[i].rotate(angle)
        
        # Horizontal flip
        if random.random() > 0.5:
            for i in range(len(image)):
                image[i] = tf.hflip(image[i])
        
        # Vertical flip  
        if random.random() > 0.5:
            for i in range(len(image)):
                image[i] = tf.vflip(image[i])
        
        img_bias_y_rate = 0.0
        img_bias_x_rate = 0.0
        scale_times = 1.0
        
        # Scale resize
        if random.random() > 0.25:
            img_h, img_w = np.array(image[0]).shape[:2]
            scale_times = torch.empty(1).uniform_(scale[0], scale[1]).item()
            scale_h, scale_w = int(img_h * scale_times), int(img_w * scale_times)
            
            # Resize RGB images (first 2) with BICUBIC
            for i in range(0, min(2, len(image))):
                image[i] = image[i].resize((scale_w, scale_h))
            
            # Resize label images with NEAREST neighbor
            for i in range(2, len(image)):
                image[i] = image[i].resize((scale_w, scale_h), resample=Image.NEAREST)
        
        if unoverlap is None:
            img_h, img_w = np.array(image[0]).shape[:2]
            img_center_x = img_w // 2
            img_center_y = img_h // 2
            
            # Calculate crop boundaries
            angle_rad = math.radians(angle)
            factor = (math.sin(angle_rad) + math.cos(angle_rad)) * scale_times
            
            crop_w_start_img1 = img_center_x - (source_size[1] * factor) // 2
            crop_w_end_img1 = img_center_x + (source_size[1] * factor) // 2 - source_size[1]
            crop_h_start_img1 = img_center_y - (source_size[0] * factor) // 2
            crop_h_end_img1 = img_center_y + (source_size[0] * factor) // 2 - source_size[0]
            
            crop_w_range_img1 = (crop_w_start_img1, crop_w_end_img1)
            crop_h_range_img1 = (crop_h_start_img1, crop_h_end_img1)
            
            crop_w_start_x_img1 = int(torch.empty(1).uniform_(min(crop_w_range_img1), max(crop_w_range_img1)).item())
            crop_h_start_y_img1 = int(torch.empty(1).uniform_(min(crop_h_range_img1), max(crop_h_range_img1)).item())
            
            # Crop all images
            for each_img in range(len(image)):
                image[each_img] = image[each_img].crop((
                    crop_w_start_x_img1,
                    crop_h_start_y_img1, 
                    crop_w_start_x_img1 + source_size[1], 
                    crop_h_start_y_img1 + source_size[0]
                ))
        else:
            # Center crop
            img_h, img_w = np.array(image[0]).shape[:2]
            img_center_x = img_w // 2
            img_center_y = img_h // 2
            crop_w_start_x = img_center_x - source_size[1] // 2
            crop_h_start_y = img_center_y - source_size[0] // 2
            
            for each_img in range(len(image)):
                image[each_img] = image[each_img].crop((
                    crop_w_start_x, 
                    crop_h_start_y, 
                    crop_w_start_x + source_size[1], 
                    crop_h_start_y + source_size[0]
                ))
        
        return image, img_bias_y_rate, img_bias_x_rate

def mirrorPadding2D(image):
    """
    Apply mirror padding to a 2D image
    
    Args:
        image: numpy array of shape (h, w, c)
    
    Returns:
        image_mirror_padding: mirror padded image of shape (2*h, 2*w, c)
    """
    h, w, c = image.shape
    th, dh = h // 2, h - h // 2
    lw, rw = w // 2, w - w // 2
    
    image_mirror_padding = np.zeros((h * 2, w * 2, c), dtype=np.uint8)
    
    # Flip operations
    image_td = np.flip(image, axis=0)
    image_lr = np.flip(image, axis=1)
    image_tdlr = np.flip(image, axis=(0, 1))
    
    # Place original image in center
    image_mirror_padding[th:(th + h), lw:(lw + w)] = image
    
    # Fill corners
    image_mirror_padding[0:th, 0:lw] = image_tdlr[dh:, rw:]
    image_mirror_padding[0:th, (lw + w):] = image_tdlr[dh:, 0:rw]
    image_mirror_padding[(dh + h):, 0:lw] = image_tdlr[0:dh, rw:]
    image_mirror_padding[(dh + h):, (lw + w):] = image_tdlr[0:dh, 0:rw]
    
    # Fill edges
    image_mirror_padding[0:th, lw:(lw + w)] = image_td[dh:, :]
    image_mirror_padding[(th + h):, lw:(lw + w)] = image_td[0:dh, :]
    image_mirror_padding[th:(th + h), 0:lw] = image_lr[:, rw:]
    image_mirror_padding[th:(th + h), (lw + w):] = image_lr[:, 0:rw]
    
    return image_mirror_padding
