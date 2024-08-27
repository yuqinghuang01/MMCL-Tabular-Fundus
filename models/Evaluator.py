from typing import Tuple

import torch
import torchmetrics
import pytorch_lightning as pl

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.GraphModel import GraphModel
from models.MultimodalModel import MultimodalModel


class Evaluator(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)
    self.batch_size = hparams.batch_size

    if self.hparams.datatype == 'imaging':
      self.model = ImagingModel(self.hparams)
    if self.hparams.datatype == 'tabular':
      self.model = TabularModel(self.hparams)
    if self.hparams.datatype == 'imaging_tabular':
      self.model = MultimodalModel(self.hparams)
    if self.hparams.datatype == 'graph_tabular':
      #self.model = GraphModel(self.hparams)
      self.model = MultimodalModel(self.hparams)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
    
    self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    
    self.pre_train = torchmetrics.Precision(task=task, num_classes=self.hparams.num_classes)
    self.pre_val = torchmetrics.Precision(task=task, num_classes=self.hparams.num_classes)
    self.pre_test = torchmetrics.Precision(task=task, num_classes=self.hparams.num_classes)
    
    self.rec_train = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes)
    self.rec_val = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes)
    self.rec_test = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes)
    
    self.f1_train = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes)
    self.f1_val = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes)
    self.f1_test = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes)

    self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    
    self.stats_val = torchmetrics.StatScores(task=task, num_classes=self.hparams.num_classes)
    self.stats_test = torchmetrics.StatScores(task=task, num_classes=self.hparams.num_classes)
    
    self.roc_val = torchmetrics.ROC(task=task, num_classes=self.hparams.num_classes)
    self.roc_test = torchmetrics.ROC(task=task, num_classes=self.hparams.num_classes)

    self.criterion = torch.nn.CrossEntropyLoss()
    
    self.best_val_score = 0
    self.best_val_acc = 0
    self.best_val_pre = 0
    self.best_val_rec = 0
    self.best_val_f1 = 0
    self.best_val_sen = 0
    self.best_val_spe = 0
    self.best_val_fpr = 0
    self.best_val_tpr = 0

    print(self.model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates a prediction from a data point
    """
    y_hat = self.model(x)

    # Needed for gradcam
    if len(y_hat.shape)==1:
      y_hat = torch.unsqueeze(y_hat, 0)

    return y_hat

  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    """
    Runs test step
    """
    if self.hparams.datatype.startswith('graph'):
      x = batch
      y = batch.label
    else:
      x, y = batch
    
    y_hat = self.forward(x)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_test(y_hat, y)
    self.pre_test(y_hat, y)
    self.rec_test(y_hat, y)
    self.f1_test(y_hat, y)
    self.auc_test(y_hat, y)
    self.stats_test(y_hat, y)
    self.roc_test(y_hat, y)

  def test_epoch_end(self, _) -> None:
    """
    Test epoch end
    """
    test_acc = self.acc_test.compute()
    test_pre = self.pre_test.compute()
    test_rec = self.rec_test.compute()
    test_f1 = self.f1_test.compute()
    test_auc = self.auc_test.compute()
    test_tp, test_fp, test_tn, test_fn, test_support = self.stats_test.compute()
    test_fpr, test_tpr, test_thresholds = self.roc_test.compute()

    self.log('test.acc', test_acc, batch_size=self.batch_size)
    self.log('test.pre', test_pre, batch_size=self.batch_size)
    self.log('test.rec', test_rec, batch_size=self.batch_size)
    self.log('test.f1', test_f1, batch_size=self.batch_size)
    self.log('test.auc', test_auc, batch_size=self.batch_size)
    self.log('test.sen', (test_tp / (test_tp + test_fn)), batch_size=self.batch_size)
    self.log('test.spe', (test_tn / (test_tn + test_fp)), batch_size=self.batch_size)
    print("test_fpr_tpr=========================================")
    torch.save((test_fpr, test_tpr), f'./rocs/{self.hparams.datatype}_{self.hparams.image_type}_{self.hparams.gnn_layer_type}_test.pt')
    
  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Train and log.
    """
    if self.hparams.datatype.startswith('graph'):
      x = batch
      y = batch.label
    else:
      x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_train(y_hat, y)
    self.auc_train(y_hat, y)

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False, batch_size=self.batch_size)

    return loss

  def training_epoch_end(self, _) -> None:
    """
    Compute training epoch metrics and check for new best values
    """
    self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train, batch_size=self.batch_size)
    self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train, batch_size=self.batch_size)

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    if self.hparams.datatype.startswith('graph'):
      x = batch
      y = batch.label
    else:
      x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_val(y_hat, y)
    self.pre_val(y_hat, y)
    self.rec_val(y_hat, y)
    self.f1_val(y_hat, y)
    self.auc_val(y_hat, y)
    self.stats_val(y_hat, y)
    self.roc_val(y_hat, y)
    
    self.log('eval.val.loss', loss, on_epoch=True, on_step=False, batch_size=self.batch_size)

    
  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return  

    epoch_acc_val = self.acc_val.compute()
    epoch_pre_val = self.pre_val.compute()
    epoch_rec_val = self.rec_val.compute()
    epoch_f1_val = self.f1_val.compute()
    epoch_auc_val = self.auc_val.compute()
    val_tp, val_fp, val_tn, val_fn, val_support = self.stats_val.compute()
    epoch_fpr_val, epoch_tpr_val, epoch_thresholds_val = self.roc_val.compute()

    self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val, batch_size=self.batch_size)
    self.log('eval.val.pre', epoch_pre_val, on_epoch=True, on_step=False, metric_attribute=self.pre_val, batch_size=self.batch_size)
    self.log('eval.val.rec', epoch_rec_val, on_epoch=True, on_step=False, metric_attribute=self.rec_val, batch_size=self.batch_size)
    self.log('eval.val.f1', epoch_f1_val, on_epoch=True, on_step=False, metric_attribute=self.f1_val, batch_size=self.batch_size)
    self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val, batch_size=self.batch_size)
    self.log('eval.val.sen', (val_tp / (val_tp + val_fn)), on_epoch=True, on_step=False, batch_size=self.batch_size)
    self.log('eval.val.spe', (val_tn / (val_tn + val_fp)), on_epoch=True, on_step=False, batch_size=self.batch_size)
  
    if self.hparams.target == 'dvm':
      self.best_val_score = max(self.best_val_score, epoch_acc_val)
    else:
      if epoch_auc_val > self.best_val_score:
        self.best_val_score = max(self.best_val_score, epoch_auc_val)
        self.best_val_acc = epoch_acc_val
        self.best_val_pre = epoch_pre_val
        self.best_val_rec = epoch_rec_val
        self.best_val_f1 = epoch_f1_val
        self.best_val_sen = (val_tp / (val_tp + val_fn))
        self.best_val_spe = (val_tn / (val_tn + val_fp))
        self.best_val_fpr = epoch_fpr_val
        self.best_val_tpr = epoch_tpr_val

    self.acc_val.reset()
    self.pre_val.reset()
    self.rec_val.reset()
    self.f1_val.reset()
    self.auc_val.reset()
    self.stats_val.reset()
    self.roc_val.reset()

  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
    return optimizer
    
    return (
      {
        "optimizer": optimizer, 
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": 'eval.val.loss',
          "strict": False
        }
      }
    )