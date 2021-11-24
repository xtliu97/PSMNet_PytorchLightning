from pytorch_lightning import Trainer

from dataloader import KITTI_lightning
from models import PSMNet_lightning

import warnings
warnings.filterwarnings('ignore')


def main():
    PATH_TO_DATASET = "../../Documents/kitti15/"
    MAX_DISP = 192
    VALIDATE_SIZE = 20
    BATCH_SIZE = 1

    model = PSMNet_lightning(MAX_DISP)
    data = KITTI_lightning(directory = PATH_TO_DATASET,
        validate_size=VALIDATE_SIZE ,
        batch_size = BATCH_SIZE ,
        num_workers =  0)

    trainer = Trainer(progress_bar_refresh_rate = 1)

    trainer.fit(model, data)

if __name__ == "__main__":
    main()