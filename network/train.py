import argparse
import os, sys

sys.path.append("..")

import torch

import numpy as np

import torch.backends.cudnn as cudnn

import os.path as osp

from MOTSDataset import MOTSDataSet, my_collate

import random
import timeit
from tensorboardX import SummaryWriter
from loss_functions import loss

from engine import Engine

from main import unet3D

import warnings

warnings.filterwarnings("ignore")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = argparse.ArgumentParser(description="Unet3D_CCQ")
    # parameters for path
    parser.add_argument("--data_dir", type=str, default='../dataset/')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/CCQ/')
    parser.add_argument("--reload_path", type=str, default='snapshots/fold1/xx.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--pretrain_path", type=str, default='MOTS_DynConv_checkpoint.pth')
    parser.add_argument("--pretrain", type=str2bool, default=False)

    # parameters for dataset
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", type=str2bool, default=False)
    parser.add_argument("--num_workers", type=int, default=1)

    # parameters for train
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--froze_epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--itrs_each_epoch", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument('--sgd', type=str2bool, default=False)
    parser.add_argument('--adam', type=str2bool, default=False)

    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_seed", type=int, default=1234)

    # parameters for unet3D
    parser.add_argument("--input_size", type=str, default='32,96,96')
    parser.add_argument("--img_dim", type=int, default=256)
    parser.add_argument("--tokens_dim", type=int, default=256)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--img_attn_layers", type=int, default=4)
    parser.add_argument("--query_attn_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_query", type=int, default=32)
    parser.add_argument("--num_cls", type=int, default=7)
    parser.add_argument("--output_channel", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop_path_rate", type=float, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--cat", type=str2bool, default=True)
    parser.add_argument("--skipattn", type=str2bool, default=True)
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = 10 * lr
    return lr


def main():
    """Create the model and start the training."""
    print('##########')
    parser = get_arguments()
    print('parser:', parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)
        model = unet3D(
            image_size=input_size,
            img_dim=args.img_dim,
            tokens_dim=args.tokens_dim,
            mlp_dim=args.mlp_dim,
            img_attn_layers=args.img_attn_layers,
            query_attn_layers=args.query_attn_layers,
            heads=args.heads,
            num_query=args.num_query,
            num_cls=args.num_cls,
            dropout=args.dropout,
            drop_path_rate=args.drop_path_rate,
            weight_std=args.weight_std,
            skipattn=args.skipattn,
            cat=args.cat

        )

        model.train()
        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)


        if args.sgd == True:
            optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)
        if args.adam == True:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.num_gpus > 1:
            model = engine.data_parallel(args, model)

        if args.FP16:
            print("Note: Using FP16 during training************")
        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.output_channel).to(device)
        loss_seg_CE = loss.CELoss4MOTS(num_classes=args.output_channel, ignore_index=255).to(device)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        trainloader, train_sampler = engine.get_train_loader(
            MOTSDataSet(args.data_dir, args.train_list, max_iters=args.itrs_each_epoch * args.batch_size,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror),
            collate_fn=my_collate)

        all_tr_loss = []

        for epoch in range(args.num_epochs):

            if epoch < args.start_epoch:
                continue
            time = []
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_Dice_loss, epoch_BCE_loss, epoch_loss = [], [], []
            adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

            for iter, batch in enumerate(trainloader):
                start = timeit.default_timer()
                images = torch.from_numpy(batch['image']).cuda()
                labels = torch.from_numpy(batch['label']).cuda()
                task_ids = batch['task_id']
                optimizer.zero_grad()
                preds = model(images, task_ids)

                term_seg_Dice = loss_seg_DICE.forward(preds, labels)
                term_seg_BCE = loss_seg_CE.forward(preds, labels)
                term_all = term_seg_Dice + term_seg_BCE

                reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                reduce_all = engine.all_reduce_tensor(term_all)
                term_all.backward()
                optimizer.step()
                epoch_Dice_loss.append(float(reduce_Dice))
                epoch_BCE_loss.append(float(reduce_BCE))
                epoch_loss.append(float(reduce_all))
                end = timeit.default_timer()
                iternal = end - start
                if (args.local_rank == 0 and args.pretrain):
                    print(
                        'Epoch {}: {}/{}, lr1 = {:.4},lr2 = {:.4}, loss_seg_Dice = {:.4}, loss_seg_BCE = {:.4}, loss_Sum = {:.4}, time={:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'],
                            optimizer.param_groups[1]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item(), iternal))
                elif args.local_rank == 0:
                    print(
                        'Epoch {}: {}/{}, lr1 = {:.4}, loss_seg_Dice = {:.4}, loss_seg_BCE = {:.4}, loss_Sum = {:.4}, time={:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item(), iternal))
                time.append(iternal)

            epoch_Dice_loss = np.mean(epoch_Dice_loss)
            epoch_BCE_loss = np.mean(epoch_BCE_loss)
            epoch_loss = np.mean(epoch_loss)
            all_tr_loss.append(epoch_loss)

            if (args.local_rank == 0 and args.pretrain):
                print(
                    'Epoch_sum {}: lr = {:.4},lr2 = {:.4}, Dice_loss = {:.4}, BCE_loss = {:.4}, loss_Sum = {:.4}, time={:.4}'.format(
                        epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], epoch_Dice_loss,
                        epoch_BCE_loss, epoch_loss.item(), np.sum(time)))
            elif args.local_rank == 0:
                print(
                    'Epoch_sum {}: lr = {:.4}, Dice_loss = {:.4}, BCE_loss = {:.4}, loss_Sum = {:.4}, time={:.4}'.format(
                        epoch, optimizer.param_groups[0]['lr'], epoch_Dice_loss,
                        epoch_BCE_loss, epoch_loss.item(), np.sum(time)))

            if (args.local_rank == 0):
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_allloss', epoch_loss.item(), epoch)
                writer.add_scalar('Train_BCE_loss', epoch_BCE_loss.item(), epoch)
                writer.add_scalar('Train_Dice_loss', epoch_Dice_loss.item(), epoch)

            if (epoch >= 0) and (args.local_rank == 0) and (
                    ((epoch % 10 == 0) and (epoch >= 1500)) or (epoch % 50 == 0)):
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir,
                                                        'MOTS_CCQ_' + args.snapshot_dir.split('/')[-2] + '_e' + str(
                                                            epoch) + '.pth'))

            if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'MOTS_CCQ_' + args.snapshot_dir.split('/')[
                    -2] + '_final_e' + str(epoch) + '.pth'))
                break


if __name__ == '__main__':
    main()
