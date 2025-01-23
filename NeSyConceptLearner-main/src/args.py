import argparse
import os
from datetime import datetime

import utils as utils

def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--mode", type=str, required=True, help="train, test, or plot")
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--l2_grads", type=float, default=1, help="Right for right reason weight"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["p2s"],
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")

    parser.add_argument("--data-dir", type=str, help="Directory to data")
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")


    
    parser.add_argument('--concept', choices=["sax", "tsfresh", "vq-vae"],
                        help='concept that should be applied to times series' )
    
    parser.add_argument(
        "--explain", action="store_true",
          help="Plot model explanations"
    )

    parser.add_argument('--n-heads', type=int,
                        help='number of set heads for set transformer')
    parser.add_argument('--set-transf-hidden', type=int,
                        help='number of hidden dimensions for set transformer')

    # SAX params
    parser.add_argument('--n-segments', default=8, type=int,
                        help='number of sax segments')
    parser.add_argument('--alphabet-size', default=4, type=int,
                        help='alphabet size for sax')

    # tsfresh params
    parser.add_argument(
        "--load-tsf", action="store_true",
          help="Load previous tsfresh data, don't run tsfresh"
    )
    parser.add_argument(
        "--filter-tsf", action="store_true",
          help="Load previous tsfresh data, don't run tsfresh"
    )
    parser.add_argument(
        "--normalize-tsf", action="store_true",
          help="Load previous tsfresh data, don't run tsfresh"
    )
    parser.add_argument('--ts-setting', choices=["fast", "mid", "slow"],
                        help='function calculator parameter for feature \
                              extraction in tsfresh' )
    
    parser.add_argument('--num-tries', default = 1, type=int,
                        help='Num of tries for avg and max deviation calculation.')
    


    args = parser.parse_args()

    # Moved the following settings to model.py
    # hard set !!!!!!!!!!!!!!!!!!!!!!!!!
    # args.n_heads = 4
    # args.set_transf_hidden = 128

    #assert args.data_dir.endswith(os.path.sep)
    #args.conf_version = args.data_dir.split(os.path.sep)[-2]
    #args.name = args.name + f"-{args.conf_version}"
    args.conf_version = "1.0"

    if args.mode == 'test' or args.mode == 'plot':
        assert args.fp_ckpt

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    utils.seed_everything(args.seed)

    return args
