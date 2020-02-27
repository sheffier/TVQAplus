import pytorch_lightning as pl
from config_new import Config
from model.stage_new import StageTrainer


def main(opt):
    # init model
    model = StageTrainer(opt)

    trainer = pl.Trainer(gpus=opt.device_ids,
                         distributed_backend=opt.distributed_backend,
                         gradient_clip_val=opt.gradient_clip_val)
    trainer.fit(model)


if __name__ == '__main__':
    parser = Config.get_parser(StageTrainer)
    opt = parser.parse()

    main(opt)
