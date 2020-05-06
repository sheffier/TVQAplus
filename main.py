import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch.backends.cudnn as cudnn
from config import Config
from model.stage import Stage
from utils import count_parameters


def main(hparams):
    # init model

    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    model = Stage(hparams)

    count_parameters(model)

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=hparams.max_es_cnt,
        verbose=True,
        mode='max'
    )

    trainer = pl.Trainer(max_epochs=hparams.n_epoch,
                         val_check_interval=hparams.val_check_interval,
                         default_root_dir=hparams.save_model_dir,
                         gpus=hparams.device_ids,
                         distributed_backend=hparams.distributed_backend,
                         early_stop_callback=early_stop_callback,
                         gradient_clip_val=hparams.gradient_clip_val,
                         progress_bar_refresh_rate=hparams.progress_bar_refresh_rate)
    trainer.fit(model)


if __name__ == '__main__':
    parser = Config().get_parser(Stage)
    hparams = parser.parse()
    hparams = Stage.verify_hparams(hparams)

    main(hparams)
