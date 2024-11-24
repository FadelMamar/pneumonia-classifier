from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from loguru import logger
from tqdm import tqdm
import pandas as pd
from PIL import Image
import pytorch_lightning as L
from torchvision import transforms
import io
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pneumonia_cls.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


class MyDataset(Dataset):

  def __init__(self,data,transform,resample:bool=False):
    self.data = data
    self.data.loc[:,"labels"] = self.data['labels'].map({0:1,1:0})
    print('labels map = {0:normal; 1:pneumo}')
    self.transform = transform

    if resample:
      self.data = self.resample(self.data)

  def resample(self,df:pd.DataFrame):

        len_ = min((df["labels"] == 0).sum(),(df["labels"] == 1).sum())
        pneumo_index = df[df["labels"] == 1].sample(n=len_,random_state=1,replace=False).index
        normal_index = df[df["labels"] == 0].sample(n=len_,random_state=1,replace=False).index
        out = pd.concat([df[df.index.isin(normal_index)],
                        df[df.index.isin(pneumo_index)]
                        ]
                        )

        return out

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    image = self.data["image"].iloc[idx]["bytes"]
    image = Image.open(io.BytesIO(image))
    label = self.data["labels"].iat[idx]
    if self.transform:
      image = self.transform(image)
    return image,label


class PediatricDataset(Dataset):

    def __init__(self,data_dir:str,split:str,transform=None,resample:bool=False):

        self.data_dir = Path(data_dir)
        self.label_map = {"PNEUMONIA":1, "NORMAL":0}
        self.transform = transform
        val_size = 0.3 # for train-val split

        if split in ['validation','train']:
            self.data_dir = self.data_dir/'train'
            pneumo_images = list((self.data_dir/'PNEUMONIA').glob('*.jpeg'))
            normal_images = list((self.data_dir/'NORMAL').glob('*.jpeg'))
            images_paths = list(zip(['PNEUMONIA']*len(pneumo_images),pneumo_images)) + list(zip(['NORMAL']*len(normal_images),normal_images))
            dict_ =  {'image_path':[j for i,j in images_paths], 
                    'labels':[self.label_map[i] for i,j in images_paths ]}
            df = pd.DataFrame.from_dict(dict_, orient='columns')
            df_train, df_val = train_test_split(df,test_size=val_size,random_state=41,shuffle=True,stratify=df['labels'])

            if split == 'train':
                self.df = df_train
            else:
                self.df = df_val
            
        elif split == 'test':
            self.data_dir = self.data_dir/'test'
            pneumo_images = list((self.data_dir/'PNEUMONIA').glob('*.jpeg'))
            normal_images = list((self.data_dir/'NORMAL').glob('*.jpeg'))
            images_paths = list(zip(['PNEUMONIA']*len(pneumo_images),pneumo_images)) + list(zip(['NORMAL']*len(normal_images),normal_images))
            dict_ =  {'image_path':[j for _,j in images_paths], 
                        'labels':[self.label_map[i] for i,_ in images_paths ]}
            self.df = pd.DataFrame.from_dict(dict_, orient='columns')
        else:
            raise NotImplementedError()
        
        if resample:
            self.df = self.resample(self.df)

    def __len__(self):
        return len(self.df)
  
    def resample(self,df:pd.DataFrame):

        len_ = min((df["labels"] == 0).sum(),(df["labels"] == 1).sum())
        pneumo_index = df[df["labels"] == 1].sample(n=len_,random_state=1,replace=False).index
        normal_index = df[df["labels"] == 0].sample(n=len_,random_state=1,replace=False).index
        out = pd.concat([df[df.index.isin(normal_index)],
                        df[df.index.isin(pneumo_index)]
                        ]
                        )

        return out

    def __getitem__(self,idx):

        image_path = self.df['image_path'].iat[idx]
        image = Image.open(image_path)
        label = self.df['labels'].iat[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.Tensor([label]).int()


class MyDataModule(L.LightningDataModule):
    def __init__(self, 
                 batchsize:int=32,
                 resample_val:bool=False,
                 resample_train:bool=False,
                 use_pediatricdataset:bool=False,pediatricdata_path:str=None):
        super().__init__()

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # flags
        self.batchsize = batchsize
        self.resample_val=resample_val
        self.resample_train=resample_train
        self.use_pediatricdataset=use_pediatricdataset
        self.pediatricdata_path=pediatricdata_path
    

    def setup(self, stage: str):

        if stage == "fit":
            if not self.use_pediatricdataset:
                self.train_data = load_dataset("trpakov/chest-xray-classification", "full", split="train",
                                            cache_dir="../data").to_pandas()
                self.val_data = load_dataset("trpakov/chest-xray-classification", "full", split="validation",
                                        cache_dir="../data").to_pandas()
                self.train = MyDataset(self.train_data, self.train_transform,resample=self.resample_train)
                self.val = MyDataset(self.val_data, self.val_transform,resample=self.resample_val)
            else:
                self.train = PediatricDataset(data_dir=self.pediatricdata_path,
                                              split='train',transform=self.train_transform,
                                              resample=self.resample_train)
                self.val = PediatricDataset(data_dir=self.pediatricdata_path,
                                            split='validation',
                                            transform=self.val_transform,
                                            resample=self.resample_val)

            print(f"train data: {len(self.train)} samples.")
            print(f"val data: {len(self.val)} samples.")

        if stage == "test":
            if not self.use_pediatricdataset:
                self.test_data = load_dataset("trpakov/chest-xray-classification", "full", split="test",
                                            cache_dir="../data").to_pandas()
                self.test = MyDataset(self.test_data, self.val_transform,resample=False)
            else:
                self.test = PediatricDataset(data_dir=self.pediatricdata_path,
                                             split='test',
                                             transform=self.val_transform,
                                             resample=False)

    def train_dataloader(self):
        train_loader = DataLoader(self.train, 
                                  batch_size=self.batchsize, 
                                  # persistent_workers=True,
                                  # num_workers=2, 
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val, 
                                batch_size=self.batchsize, 
                                # persistent_workers=True,
                                # num_workers=2, 
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test, 
                                 batch_size=self.batchsize,
                                  # num_workers=2, 
                                 shuffle=False)
        return test_loader

