import imgaug.augmenters as iaa
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random

#updata nov 6:return query,reference,aug_reference
def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
                
    return data

class U1652DatasetTrain(Dataset):
    
    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5, prob_weather=.5,
                 shuffle_batch_size=32):
        super().__init__()
 

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        
        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        
        self.pairs = []
        
        for idx in self.ids:
            
            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            
            for g in gallery_imgs:
                self.pairs.append((idx, query_img, "{}/{}".format(gallery_path, g)))
        
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.prob_weather = prob_weather
        self.shuffle_batch_size = shuffle_batch_size
        self.saved_count = 0  # Counter to keep track of saved images
        
        self.samples = copy.deepcopy(self.pairs)

        # Define augmentation sequences for each weather condition
        self.augmentation_sequences = [
            iaa.Fog(),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.05, 0.2)),
            iaa.Snowflakes(density=(0.01, 0.075), flake_size=(0.3, 0.8), speed=(0.008, 0.03)),
            iaa.Sequential([iaa.Fog(), iaa.Rain()]),
            iaa.Sequential([iaa.Fog(), iaa.Snowflakes()]),
            iaa.Sequential([iaa.Rain(), iaa.Snowflakes()]),
            iaa.Multiply((0.2, 0.6)),
            iaa.Multiply((1.5, 2)),
            iaa.MotionBlur(k=(7, 10), angle=[-90, 90])
        ]


    def __getitem__(self, index):
        
        idx, query_img_path, gallery_img_path = self.samples[index]
        
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1) 

        # Apply one of the augmentations uniformly at random to the gallery image
        if np.random.random() < self.prob_weather:
            aug = random.choice(self.augmentation_sequences)
            aug_query_img = aug(image=query_img)


            # Save augmented gallery images for checking
            if self.saved_count < 3:  # Save only 3 images
                print("img_path:",query_img_path)
                save_path = os.path.join("/gpfs2/scratch/ychen57", f"aug_query_{self.saved_count}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(aug_query_img, cv2.COLOR_RGB2BGR))
                self.saved_count += 1
               
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            aug_query_img = self.transforms_query(image = aug_query_img)['image']
            
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, aug_query_img,gallery_img, idx

    def __len__(self):
        return len(self.samples)
    
    
    def shuffle(self, ):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
              
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
            idx_batch = set()
     
            # buckets
            batches = []
            current_batch = []
             
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    idx, _, _ = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
            
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 512:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  
    
        
        
class U1652DatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()
 

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms
        self.augmentation_sequences = [
            iaa.Fog(),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.05, 0.2)),
            iaa.Snowflakes(density=(0.01, 0.075), flake_size=(0.3, 0.8), speed=(0.008, 0.03)),
            iaa.Sequential([iaa.Fog(), iaa.Rain()]),
            iaa.Sequential([iaa.Fog(), iaa.Snowflakes()]),
            iaa.Sequential([iaa.Rain(), iaa.Snowflakes()]),
            iaa.Multiply((0.2, 0.6)),
            iaa.Multiply((1.5, 2)),
            iaa.MotionBlur(k=(7, 10), angle=[-90, 90])
        ]
        self.given_sample_ids = sample_ids
        
        self.images = []
        self.sample_ids = []
        
        self.mode = mode
        self.saved_count = 0
        
        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
                
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                    
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                      file))
                
                self.sample_ids.append(sample_id) 
                    
  
            
        
        
    def __getitem__(self, index):
        img_path = self.images[index]
        sample_id = self.sample_ids[index]
        
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if self.saved_count ==0 or self.saved_count ==3 or self.saved_count ==4:  # Save only 3 images
        #     print("img_path:",img_path)
        #     save_path = os.path.join("/gpfs2/scratch/ychen57", f"origin_{self.saved_count}.jpg")
        #     cv2.imwrite(save_path, img)
        aug = random.choice(self.augmentation_sequences)

        # if self.saved_count ==0 or self.saved_count ==3 or self.saved_count ==4:  # Save only 3 images
        #     snow_img = self.snow(image=img)
        #     rain_img = self.rain(image=img)
        #     save_snow_path = os.path.join("/gpfs2/scratch/ychen57", f"gallery_sonw_{self.saved_count}.jpg")
        #     save_rain_path = os.path.join("/gpfs2/scratch/ychen57", f"gallery_rain_{self.saved_count}.jpg")
        #     cv2.imwrite(save_snow_path, cv2.cvtColor(snow_img, cv2.COLOR_RGB2BGR))
        #     cv2.imwrite(save_rain_path, cv2.cvtColor(rain_img, cv2.COLOR_RGB2BGR))
        # self.saved_count += 1
        img = aug(image=img)

        # Save augmented gallery images for checking


            # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1
        
        return img, label

    def __len__(self):
        return len(self.images)
    
    def get_sample_ids(self):
        return set(self.sample_ids)
    
    
def                     get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
                                
                             
                                
    
    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*img_size[0]),
                                                               max_width=int(0.2*img_size[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*img_size[0]),
                                                               min_width=int(0.1*img_size[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])
    
    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                        A.OneOf([
                                                 A.AdvancedBlur(p=1.0),
                                                 A.Sharpen(p=1.0),
                                              ], p=0.3),
                                        A.OneOf([
                                                 A.GridDropout(ratio=0.4, p=1.0),
                                                 A.CoarseDropout(max_holes=25,
                                                                 max_height=int(0.2*img_size[0]),
                                                                 max_width=int(0.2*img_size[0]),
                                                                 min_holes=10,
                                                                 min_height=int(0.1*img_size[0]),
                                                                 min_width=int(0.1*img_size[0]),
                                                                 p=1.0),
                                              ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])
    
    return val_transforms, train_sat_transforms, train_drone_transforms