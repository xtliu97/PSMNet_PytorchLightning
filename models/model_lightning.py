from models.PSMNet import PSMNet
import pytorch_lightning as pl
from models.smoothloss import SmoothL1Loss
import torch.optim as optim
import torch


class PSMNet_lightning(pl.LightningModule):
    def __init__(self, max_disp, learning_rate = 0.001):
        super().__init__()
        self.model = PSMNet(max_disp)
        self.loss = SmoothL1Loss()
        self.max_disp = max_disp
        self.lr = learning_rate

    def forward(self, left, right):
        return self.model(left, right)
    
    def configure_optimizers(self):
      optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
      return optimizer
    
    def training_step(self, batch, batch_idx):
        left_img = batch['left']
        right_img = batch['right']
        target_disp = batch['disp']

        mask = (target_disp > 0 )& (target_disp < self.max_disp) 
        mask = mask.detach_()

        disp1, disp2, disp3 = self(left_img, right_img)
        loss1, loss2, loss3 = self.loss(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
        total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

        self.log('training_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    def training_epoch_start(self,):
        print("current lr: ", self.lr)
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if epoch == 200:
            self.lr = 0.0001
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr
        # update params
        optimizer.step(closure=optimizer_closure)
    
            
    def validation_step(self, batch, batch_idx):
        avg_error = 0.0

        left_img = batch['left']
        right_img = batch['right']
        target_disp = batch['disp']

        mask = (target_disp > 0 )& (target_disp < self.max_disp) 
        mask = mask.detach_()
        _, _, disp = self(left_img, right_img)

        #calculate 3px error
        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (((delta >= 3.0) + (delta >= 0.05 * (target_disp[mask]))) == 2)
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100
        return error


    def validation_epoch_end(self, validation_step_outputs):
        mean_error = sum(validation_step_outputs) / len(validation_step_outputs)
        self.log('3px_error', mean_error, on_step=False, on_epoch=True, prog_bar=True)
    