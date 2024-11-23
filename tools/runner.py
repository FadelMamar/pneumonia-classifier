
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
from PIL import Image
import pytorch_lightning as L
import io
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
import wandb
from functools import partial
from pytorch_lightning.callbacks  import EarlyStopping,ModelCheckpoint
import ast

class PulmonieData(Dataset):

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
                        df[df.index.isin(pneumo_index)]]
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

class PulmonieDataModule(L.LightningDataModule):
    def __init__(self, batchsize:int=32,resample_val:bool=False,resample_train:bool=False):
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
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224))
        ])
        self.batchsize = batchsize
        self.resample_val=resample_val
        self.resample_train=resample_train
    

    def setup(self, stage: str):
        self.train_data = load_dataset("trpakov/chest-xray-classification", "full", split="train").to_pandas()
        self.test_data = load_dataset("trpakov/chest-xray-classification", "full", split="test").to_pandas()
        self.val_da = load_dataset("trpakov/chest-xray-classification", "full", split="validation").to_pandas()
        self.val_ajout = pd.read_csv("/teamspace/studios/this_studio/pneumo_test.csv")
        def safe_eval(text):
            try:
                return ast.literal_eval(text)
            except Exception:
                return None  # Ou une valeur par défaut si la conversion échoue

        self.val_ajout["image"] = self.val_ajout["image"].apply(safe_eval)

        self.val_data = pd.concat([self.val_da,  self.val_ajout], ignore_index=True)


        if stage == "fit":
            self.train = PulmonieData(self.train_data, self.train_transform,resample=self.resample_train)
            self.val = PulmonieData(self.val_data, self.val_transform,resample=self.resample_val)

            print(f"train data: {len(self.train)} samples.")
            print(f"val data: {len(self.val)} samples.")

        if stage == "test":
            self.test = PulmonieData(self.test_data, self.val_transform,resample=False)

    def train_dataloader(self):
        train_loader = DataLoader(self.train, batch_size=self.batchsize, num_workers=2, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val, batch_size=self.batchsize, num_workers=2, shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test, batch_size=self.batchsize, num_workers=2, shuffle=False)
        return test_loader

class Orchestrator(L.LightningModule):
  def __init__(self, pos_weight, lr):
    super().__init__()
    self.save_hyperparameters()
    self.pos_weight = pos_weight
    self.lr = lr

    self.model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
    self.model.classifier = torch.nn.Linear(self.model.classifier.in_features,1)

    self.train_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
    self.val_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
    self.test_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)

    self.val_precision = torchmetrics.Precision(task="binary", threshold=0.5)
    self.test_precision = torchmetrics.Precision(task="binary", threshold=0.5)

    self.val_recall = torchmetrics.Recall(task="binary", threshold=0.5)
    self.test_recall = torchmetrics.Recall(task="binary", threshold=0.5)

    self.val_f1 = torchmetrics.F1Score(task="binary", threshold=0.5)
    self.test_f1 = torchmetrics.F1Score(task="binary", threshold=0.5)

    self.val_auc_pr = torchmetrics.AveragePrecision(task="binary")
    self.test_auc_pr = torchmetrics.AveragePrecision(task="binary")

    self.val_auc_roc = torchmetrics.AUROC(task="binary")
    self.test_auc_roc = torchmetrics.AUROC(task="binary")

    self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)
    self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)

    self.valpreds = []
    self.valtrue = []


  def forward(self,x):
    return self.model(x)



  def shared_step(self,batch,stage:str,batch_idx):
    image,labels = batch
    # print(f"Predictions: {labels}")
    # print(f"Predictions: {labels.shape}")
    labels = labels.float().unsqueeze(1)
   # print(f"apres: {labels}")
    logits = self(image)

    loss = self.loss_fn(logits,labels)
    preds = torch.sigmoid(logits)


    if stage == "train":
      self.train_accuracy(preds,labels.int())
      self.log("train_loss",loss, on_epoch=True ,prog_bar=True)
      self.log("train_acc",self.train_accuracy, on_epoch=True , prog_bar=True)
      return loss

    if stage == "val":
    # Mise à jour des métriques
      self.val_accuracy(preds, labels.int())
      self.val_precision(preds, labels.int())
      self.val_recall(preds, labels.int())
      self.val_f1(preds, labels.int())
      self.val_auc_pr(preds, labels.int())
      self.val_auc_roc(preds, labels.int())
      #self.val_confusion_matrix.update(preds, labels.int())

      # Log des métriques
      self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
      self.log("val_precision", self.val_precision, on_epoch=True, prog_bar=True)
      self.log("val_recall", self.val_recall, on_epoch=True, prog_bar=True)
      self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
      self.log("val_auc_pr", self.val_auc_pr, on_epoch=True, prog_bar=True)
      self.log("val_auc_roc", self.val_auc_roc, on_epoch=True, prog_bar=True)
      self.log("val_loss", loss, prog_bar=True)

      # Calculer et loguer la matrice de confusion
      #conf_matrix = self.val_confusion_matrix.compute().cpu().numpy()

      # Visualisation avec Seaborn
      #
      self.valpreds.append((torch.sigmoid(logits).cpu()>0.5).float())
      self.valtrue.append(labels.int().cpu())

      return logits , labels

    if stage == "test":

      self.test_accuracy(preds,labels.int())
      self.test_precision(preds,labels.int())
      self.test_recall(preds,labels.int())
      self.test_f1(preds,labels.int())
      self.test_auc_pr(preds,labels.int())
      self.test_auc_roc(preds,labels.int())

      self.log("test_accuracy",self.test_accuracy, prog_bar=True)
      self.log("test_precision",self.test_precision, prog_bar=True)
      self.log("test_recall",self.test_recall, prog_bar=True)
      self.log("test_f1",self.test_f1, prog_bar=True)
      self.log("test_auc_pr",self.test_auc_pr, prog_bar=True)
      self.log("test_loss",loss)

      return loss


  def training_step(self,batch,batch_idx):
    return self.shared_step(batch,"train",batch_idx)

  def validation_step(self,batch,batch_idx):
    return self.shared_step(batch,"val",batch_idx)

  def test_step(self,batch,batch_idx):
    return self.shared_step(batch,"test",batch_idx)

  def on_validation_epoch_end(self):

    # get preds and labels
    preds = torch.cat(self.valpreds).numpy().flatten().tolist()
    labels = torch.cat(self.valtrue).numpy().flatten().tolist()

    #print(preds.shape,labels.shape)
    if self.current_epoch % 10 == 0:
      wandb.log({f"conf_mat_{self.current_epoch}" : wandb.plot.confusion_matrix(preds=preds,
                          y_true=labels,
                          class_names=["normale","pneumonie"])})
    # clear
    self.valpreds.clear()
    self.valtrue.clear()



  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
    return optimizer


if __name__ == "__main__":


    wandb_logger = WandbLogger(project='pneumonia-detection_densenet121', name = "AjoutDonnee_val",log_model=True)

    data =  PulmonieDataModule(batchsize=64,
                                resample_val=True,
                                resample_train=False)

    orch = Orchestrator(pos_weight=torch.tensor([2.0]),lr=1e-3)
    early_stop_callback = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=10, verbose=False, mode="max")
    checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="val_f1",
    mode="max",
    dirpath="/teamspace/studios/this_studio/checkpoint/",
    filename="sample-mnist-{epoch:02d}-{val_f1:.2f}",
)
    callbacks = [early_stop_callback,checkpoint_callback]

    


    trainer = L.Trainer(max_epochs=50, 
                        accelerator='gpu', 
                        logger=wandb_logger,
                        precision='16-mixed',
                        deterministic=False,
                        reload_dataloaders_every_n_epochs=3,
                        callbacks = callbacks
                        )
    #trainer.fit(orch, data)


    trainer.test(orch,data,ckpt_path="/teamspace/studios/this_studio/checkpoint/sample-mnist-epoch=04-val_f1=0.71.ckpt")

    dummy_input = torch.randn(1, 3, 224, 224)

    import torch.onnx

# Exportation du modèle vers ONNX
onnx_file_path = "pulmonie.onnx"
torch.onnx.export(orch,              # Le modèle PyTorch
                  dummy_input,        # Exemple de donnée (batch de taille 1, image 224x224)
                  onnx_file_path,     # Chemin où enregistrer le fichier ONNX
                  export_params=True, # Enregistrer les poids du modèle
                  opset_version=12,   # Version de l'API ONNX
                  do_constant_folding=True,  # Optimisation de constante
                  input_names=['input'],  # Noms des entrées
                  output_names=['output'],  # Noms des sorties
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # Pour des tailles de batch dynamiques