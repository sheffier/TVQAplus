import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch.backends.cudnn as cudnn
from config_new import Config
from model.stage_new import StageTrainer
from utils import count_parameters


def main(opt):
    # init model

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    model = StageTrainer(opt)

    count_parameters(model)

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=opt.max_es_cnt,
        verbose=True,
        mode='max'
    )

    trainer = pl.Trainer(default_save_path=opt.save_model_dir,
                         gpus=opt.device_ids,
                         distributed_backend=opt.distributed_backend,
                         early_stop_callback=early_stop_callback,
                         gradient_clip_val=opt.gradient_clip_val)
    trainer.fit(model)


if __name__ == '__main__':
    parser = Config().get_parser(StageTrainer)
    opt = parser.parse()

    main(opt)
