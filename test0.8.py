import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np
from sample4geo.dataset.university import U1652DatasetEval, get_transforms, CustomDataset
from sample4geo.model import TimmModel
from sample4geo.trainer import predict



@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 16
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all or int
    
    # Dataset
    dataset: str = 'U1652-D2S'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"
    
    # Checkpoint to start from
    checkpoint_start = 'university/convnext_base.fb_in22k_ft_in1k_384/102030/weights_end.pth'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'U1652-D2S':
    config.query_folder_train = '/gpfs2/scratch/xzhang31/university-1652/University-1652/train/satellite'
    config.gallery_folder_train = '/gpfs2/scratch/xzhang31/university-1652/University-1652/train/drone'   
    config.query_folder_test = '/gpfs2/scratch/xzhang31/university-1652/University-1652WX/query_drone160k_wx/query_drone_160k_wx_24' 
    config.gallery_folder_test = '/gpfs2/scratch/xzhang31/university-1652/University-1652WX/gallery_satellite_160k'   
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = './data/U1652/train/satellite'
    config.gallery_folder_train = './data/U1652/train/drone'    
    config.query_folder_test = './data/U1652/test/query_satellite'
    config.gallery_folder_test = './data/U1652/test/gallery_drone'


def predict_top_k(query_features, gallery_features, k=10):
    similarities = torch.matmul(query_features, gallery_features.t())
    top_k_indices = torch.argsort(similarities, descending=True)[:, :k]
    return top_k_indices

if __name__ == '__main__':
    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nModel: {}".format(config.model))

    model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, _, _ = get_transforms(img_size, mean=mean, std=std)
    
    # Read the query order from the Query TXT file
    with open('query_drone_name.txt', 'r') as query_file:
        query_order = [line.strip() for line in query_file.readlines()]
    
    # Reference Satellite Images
    '''query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                          mode="query",
                                          transforms=val_transforms)'''

    query_dataset_test = CustomDataset(data_folder=config.query_folder_test, image_files=query_order, transforms=val_transforms)
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=False)
    
    # Query Ground Images Test
    '''gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                            mode="gallery",
                                            transforms=val_transforms,
                                            sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n)'''

    gallery_dataset_test = CustomDataset(data_folder=config.gallery_folder_test, transforms=val_transforms)

    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=False)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # Extract features
    print("Extract Features:")
    query_features, ids_query = predict(config, model, query_dataloader_test)
    gallery_features, ids_gallery = predict(config, model, gallery_dataloader_test)



    # Predict top 10
    top_10_indices = predict_top_k(query_features, gallery_features, k=10)

    # Save results to file
    gallery_filenames = np.array(gallery_dataset_test.image_files)
    gallery_filenames_no_ext = np.array([os.path.splitext(filename)[0] for filename in gallery_filenames])

    with open('answer08.txt', 'w') as f:
        for indices in top_10_indices:
            top_10_filenames = '\t'.join(gallery_filenames_no_ext[indices].tolist())
            f.write(f"{top_10_filenames}\n")

    print("Top 10 predictions saved to answer08.txt")