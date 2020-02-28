import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from config_new import Config
from model.stage_new import StageTrainer


def main(opt):
    # init model
    model = StageTrainer(opt)

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=10,
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
