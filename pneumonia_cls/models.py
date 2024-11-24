import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms
import torchvision
from PIL import Image
import pytorch_lightning as L
import torchmetrics
import wandb


class Classifier(object):
  
    def __init__(self, model_path:str, threshold:float=0.5,device:str='cpu'):

        self.threshold=threshold
        self._load_model(model_path)
        self.model=None
        self.modeltype=None
        self.device=device

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    #TODO
    def _load_model(self,model_path:str):

        if ".onnx" in model_path:
            self.modeltype = "onnx"
            self.model = ... # TODO load onnx runtime

        elif ".pt" in model_path:
            self.modeltype = 'pt'
            self.model = torch.load(model_path,map_location=torch.device('cpu'),weights_only=True)
            self.model = self.model.to(self.device)

        else:
            raise NotImplementedError()
    
    
    def __call__(self, image:Image.Image):
        
        in_ = self.pre_process(image)

        probs = self.perform_inference(in_)

        out = self.post_process(probs)

        return out
    
    def perform_inference(self,image:torch.Tensor|np.ndarray):
       
        if self.modeltype == 'onnx':
            return torch.from_numpy(self.model(image)).sigmoid()
        elif self.modeltype == 'pt':
            return self.model(image.to(self.device)).sigmoid()
    
    def pre_process(self,image:Image.Image):
       
       if self.modeltype == 'onnx':
            return self.transform(image).cpu().numpy()
       
       elif self.modeltype == 'pt':
            return self.transform(image)
       
    def post_process(self,probs:torch.Tensor):
       return (probs > self.threshold).int().cpu().numpy()


class Orchestrator(L.LightningModule):
  def __init__(self, pos_weight, lr):
    super().__init__()
    self.save_hyperparameters()
    self.lr = lr

    self.model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    # self.model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
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

    # self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)
    # self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)

    self.valpreds = []
    self.valtrue = []

  def forward(self,x):
    return self.model(x)

  def shared_step(self,batch,stage:str,batch_idx):
    image,labels = batch
    labels = labels.float().unsqueeze(1)
    logits = self(image)

    loss = self.loss_fn(logits,labels)
    probs = torch.sigmoid(logits)

    if stage == "train":
      self.train_accuracy(probs,labels.int())
      self.log("train_loss",loss, on_epoch=True ,prog_bar=True)
      self.log("train_acc",self.train_accuracy, on_epoch=True ,prog_bar=True)

    elif stage == "val":
    # Mise à jour des métriques
      self.val_accuracy(probs, labels.int())
      self.val_precision(probs, labels.int())
      self.val_recall(probs, labels.int())
      self.val_f1(probs, labels.int())
      self.val_auc_pr(probs, labels.int())
      self.val_auc_roc(probs, labels.int())

      # Log des métriques
      self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=False)
      self.log("val_precision", self.val_precision, on_epoch=True, prog_bar=False)
      self.log("val_recall", self.val_recall, on_epoch=True, prog_bar=False)
      self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
      self.log("val_auc_pr", self.val_auc_pr, on_epoch=True, prog_bar=False)
      self.log("val_auc_roc", self.val_auc_roc, on_epoch=True, prog_bar=False)
      self.log("val_loss", loss, prog_bar=True)

    # recording
      self.valpreds.append((probs.cpu()>0.5).float())
      self.valtrue.append(labels.int().cpu())

      # return logits , labels

    elif stage == "test":

      self.test_accuracy(probs,labels.int())
      self.test_precision(probs,labels.int())
      self.test_recall(probs,labels.int())
      self.test_f1(probs,labels.int())
      self.test_auc_pr(probs,labels.int())
      self.test_auc_roc(probs,labels.int())

      self.log("test_accuracy",self.test_accuracy, prog_bar=True, on_epoch=True)
      self.log("test_precision",self.test_precision, prog_bar=True,on_epoch=True)
      self.log("test_recall",self.test_recall, prog_bar=True,on_epoch=True)
      self.log("test_f1",self.test_f1, prog_bar=True,on_epoch=True)
      self.log("test_auc_pr",self.test_auc_pr, prog_bar=True,on_epoch=True)
      self.log("test_loss",loss)
    
    else:
       raise NotImplementedError()

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

