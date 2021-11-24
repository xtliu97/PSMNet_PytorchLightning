import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataloader.KITTI2015_loader import KITTI2015, RandomCrop, ToTensor, Normalize, Pad

class KITTI_lightning(pl.LightningDataModule):

    def __init__(self, directory, validate_size=20, batch_size = 2, num_workers = 8, occ=True):
        super().__init__()
        self.directory = directory
        self.validate_size = validate_size
        self.occ = occ
        self.batch_size = batch_size
        self.num_workers = num_workers

        mean = [0.406, 0.456, 0.485]
        std = [0.225, 0.224, 0.229]
        self.train_transform = T.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])
        self.validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])

    def setup(self, stage=None):
        # 这里实现training step中使用的trainset和valset
        if stage == 'fit' or stage is None:
            self.trainset = KITTI2015(self.directory, mode='train', validate_size = self.validate_size, transform=self.train_transform)
            self.valset = KITTI2015(self.directory, mode='validate',validate_size = self.validate_size, transform=self.validate_transform)
        
        else:
            raise NotImplementedError
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, num_workers=self.num_workers, shuffle=False)
    
    """
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    """