from __future__ import annotations
from argparse import ArgumentParser


class Config:
    def __init__(self):
        self.parser = None

    def get_parser(self, model_cls) -> Config:
        self.parser = ArgumentParser(description=f"parser for {model_cls.__name__} model")

        self.parser.add_argument("--save-model-dir", type=str, default=None,
                                 help="path to folder where trained model will be saved.")
        self.parser.add_argument("--checkpoint-model-dir", type=str, default="",
                                 help="path to folder where checkpoints of trained models will be saved")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument('--distributed-backend', type=str, default='', choices=('dp', 'ddp', 'ddp2'),
                                 help='supports three options dp, ddp, ddp2')
        self.parser.add_argument("--num_workers", type=int, default=2,
                                 help="num subprocesses used to load the data, 0: use main process")
        self.parser.add_argument("--seed", type=int, default=2018,
                                 help="random seed")
        self.parser.add_argument("--val_check_interval", type=float, default=1.0,
                                 help="(float|int): How often within one training epoch to check the validation set."
                                      "If float, '%' of tng epoch. If int, check every n batch")
        self.parser.add_argument("--checkpoint-interval", type=int, default=-1,
                                 help="number of batches after which a checkpoint of the trained model will be created."
                                      " When set to -1 (default) checkpoint is disabled")
        self.parser.add_argument("--progress_bar_refresh_rate", type=int, default=1,
                                 help="How often to refresh progress bar (in steps).")
        self.parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=3e-7, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=5, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=16, help="mini-batch size for testing")

        self.parser = model_cls.add_model_specific_args(self.parser)

        return self

    def parse(self):
        hparams = self.parser.parse_args()

        # if hparams.distributed_backend != '' and hparams.distributed_backend == 'ddp':
        #     hparams.num_workers = 0

        if hparams.val_check_interval.is_integer() and hparams.val_check_interval > 1:
            hparams.val_check_interval = int(hparams.val_check_interval)
        else:
            if not 0. <= hparams.val_check_interval <= 1.:
                msg = f"`val_check_interval` must lie in the range [0.0, 1.0], but got {hparams.val_check_interval:.3f}."
                raise ValueError(msg)

        return hparams
