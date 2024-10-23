import os
import os.path as osp
import argparse
from dataloader.liveness_dataloader import get_dataset_loader, get_tgt_dataset_loader
from misc.util import init_model, init_random_seed
from misc.saver import Saver
import warnings

warnings.filterwarnings("ignore")


def reload(net, restore):
    import torch
    state_dict = torch.load(restore)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)


def main(args):

    import time
    args.path = str(time.strftime("%Y%m%d%H%M")) + "_" + args.path

    t = int(time.time()) % 100000
    args.manual_seed = t

    args.seed = init_random_seed(args.manual_seed)

    if args.setting == "C":
        args.dataset1 = 'OULU'
        args.dataset2 = 'MSU'
        args.dataset3 = 'idiap'
        args.dataset_target = 'CASIA'
    elif args.setting == "M":
        args.dataset1 = 'idiap'
        args.dataset2 = 'OULU'
        args.dataset3 = 'CASIA'
        args.dataset_target = 'MSU'
    elif args.setting == "O":
        args.dataset1 = 'CASIA'
        args.dataset2 = 'idiap'
        args.dataset3 = 'MSU'
        args.dataset_target = 'OULU'
    elif args.setting == "I":
        args.dataset1 = 'OULU'
        args.dataset2 = 'CASIA'
        args.dataset3 = 'MSU'
        args.dataset_target = 'idiap'
    else:
        raise Exception("Wrong setting!")

    args.results_path = './result/results_' + args.path

    saverootname = args.dataset1 + '_' + args.dataset2 + '_' + args.dataset3 + '_to_' + args.dataset_target + '-batchsize' + str(
        args.batchsize)
    savelogname = osp.join(saverootname, 'train')
    snapfilepath = osp.join(args.results_path, saverootname, 'snapshots')

    os.makedirs(snapfilepath, exist_ok=True)

    saver = Saver(args, savelogname)
    saver.print_config()

    print(args.results_path)

    data_loader1_real = get_dataset_loader(name=args.dataset1, getreal=True, batch_size=args.batchsize, args=args, in_channels=3)
    data_loader1_fake = get_dataset_loader(name=args.dataset1, getreal=False, batch_size=args.batchsize, args=args, in_channels=3)

    data_loader2_real = get_dataset_loader(name=args.dataset2, getreal=True, batch_size=args.batchsize, args=args, in_channels=3)
    data_loader2_fake = get_dataset_loader(name=args.dataset2, getreal=False, batch_size=args.batchsize, args=args, in_channels=3)

    data_loader3_real = get_dataset_loader(name=args.dataset3, getreal=True, batch_size=args.batchsize, args=args, in_channels=3)
    data_loader3_fake = get_dataset_loader(name=args.dataset3, getreal=False, batch_size=args.batchsize, args=args, in_channels=3)

    data_loader_target = get_tgt_dataset_loader(args=args, name=args.dataset_target, batch_size=args.batchsize_test, in_channels=3, getreal=None)
    from models.MADDG import Encoder

    model = Encoder(in_channels=3, mode=args.norm)
    resume_path = './result/results_' + args.resume_path
    resume_saverootname = args.dataset1 + '_' + args.dataset2 + '_' + args.dataset3 + '_to_' + args.dataset_target + '-batchsize' + str(
        args.resume_bs) 

    epoch = None
    step = None

    from core.train import Train
    Train(args, model,
              data_loader1_real, data_loader1_fake,
              data_loader2_real, data_loader2_fake,
              data_loader3_real, data_loader3_fake,
              data_loader_target,
              saver, saverootname, epoch, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HPDR")

    parser.add_argument('--setting', type=str, default='M')

    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--metatrainsize', type=int, default=2)

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--meta_step_size', type=float, default=1e-3)

    parser.add_argument('--hter', type=float, default=0.12)
    parser.add_argument('--auc', type=float, default=0.9)

    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--batchsize_test', type=int, default=30)

    parser.add_argument('--image_size', type=int, default=256)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--model_save_step', type=int, default=10)
    parser.add_argument('--model_save_epoch', type=int, default=1)

    parser.add_argument('--norm', type=str, default="IN")
    
    parser.add_argument('--path', type=str, default="SET_YOUR_SAVE_PATH")
    print(parser.parse_args())
    main(parser.parse_args())
