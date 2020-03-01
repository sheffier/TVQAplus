from __future__ import annotations
from argparse import ArgumentParser


class Config:
    def __init__(self):
        self.parser = None

    def get_parser(self, model_cls) -> Config:
        self.parser = ArgumentParser(description=f"parser for {model_cls.__name__} model")

        self.parser.add_argument("--save-model-dir", type=str, required=True,
                                 help="path to folder where trained model will be saved.")
        self.parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                 help="path to folder where checkpoints of trained models will be saved")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument('--distributed-backend', type=str, default=None, choices=('dp', 'ddp', 'ddp2'),
                                 help='supports three options dp, ddp, ddp2')
        self.parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
        self.parser.add_argument("--num_workers", type=int, default=2,
                                 help="num subprocesses used to load the data, 0: use main process")
        self.parser.add_argument("--seed", type=int, default=2018,
                                 help="random seed")
        self.parser.add_argument("--log-interval", type=int, default=800,
                                 help="number batches which the training loss is logged, default is 800")
        self.parser.add_argument("--checkpoint-interval", type=int, default=-1,
                                 help="number of batches after which a checkpoint of the trained model will be created."
                                      " When set to -1 (default) checkpoint is disabled")

        self.parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=3e-7, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=5, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=16, help="mini-batch size for testing")

        self.parser = model_cls.add_model_specific_args(self.parser)

        return self

    def parse(self):
        opt = self.parser.parse_args()
        opt.vfeat_flag = "vfeat" in opt.input_streams
        opt.sub_flag = "sub" in opt.input_streams
        opt.concat_ctx = len(opt.input_streams) == 2

        if opt.distributed_backend is not None and opt.distributed_backend == 'ddp':
            opt.num_workers = 0

        return opt
