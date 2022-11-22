import argparse
import os, sys

sys.path.append("..")

import torch
import torch.nn as nn

import numpy as np

import torch.backends.cudnn as cudnn

from scipy.ndimage.filters import gaussian_filter




from main import unet3D
from MOTSDataset import MOTSValDataSet

import timeit

import nibabel as nib
from math import ceil

from engine import Engine

start = timeit.default_timer()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MOTS: CQTr solution!")

    parser.add_argument("--data_dir", type=str, default='../dataset/')
    parser.add_argument("--val_list", type=str, default='list/MOTS/tt.txt')
    parser.add_argument("--reload_path", type=str, default='snapshots/fold1/MOTS_DynConv_fold1_final_e999.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--save_path", type=str, default='outputs/')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

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


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def multi_net(net_list, img, task_id):


    padded_prediction = net_list[0](img, task_id)
    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img, task_id)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction


def predict_sliding(args, net_list, image, tile_size, classes, task_id):
    gaussian_importance_map = _get_gaussian(tile_size, sigma_scale=1. / 8)

    image_size = image.shape
    overlap = 1 / 2

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3], image_size[4]))
    count_predictions = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3], image_size[4]))
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img = torch.from_numpy(img).cuda()

                # ----------------------------- Gather Batch ------------------------------------
                x_list = [img, torch.flip(img, [2]), torch.flip(img, [3]), torch.flip(img, [4]),
                          torch.flip(img, [2, 3]), torch.flip(img, [2, 4]), torch.flip(img, [3, 4]),
                          torch.flip(img, [2, 3, 4])]
                x = torch.cat(x_list, 0)

                pre = multi_net(net_list, x, task_id.repeat(x_list.__len__()))
                prediction1 = pre[0:1]
                prediction2 = torch.flip(pre[1:2], [2])
                prediction3 = torch.flip(pre[2:3], [3])
                prediction4 = torch.flip(pre[3:4], [4])
                prediction5 = torch.flip(pre[4:5], [2, 3])
                prediction6 = torch.flip(pre[5:6], [2, 4])
                prediction7 = torch.flip(pre[6:7], [3, 4])
                prediction8 = torch.flip(pre[7:8], [2, 3, 4])

                # ------------------------ one-by-one ---------------------------------
                # prediction1 = multi_net(net_list, img, task_id)
                # prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2]), task_id), [2])
                # prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3]), task_id), [3])
                # prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [4]), task_id), [4])
                # prediction5 = torch.flip(multi_net(net_list, torch.flip(img, [2,3]), task_id), [2,3])
                # prediction6 = torch.flip(multi_net(net_list, torch.flip(img, [2,4]), task_id), [2,4])
                # prediction7 = torch.flip(multi_net(net_list, torch.flip(img, [3,4]), task_id), [3,4])
                # prediction8 = torch.flip(multi_net(net_list, torch.flip(img, [2,3,4]), task_id), [2,3,4])
                prediction = (
                                         prediction1 + prediction2 + prediction3 + prediction4 + prediction5 + prediction6 + prediction7 + prediction8) / 8.
                prediction = prediction.cpu()

                prediction[:, :] *= gaussian_importance_map

                if isinstance(prediction, list):
                    shape = np.array(prediction[0].shape)
                    shape[0] = prediction[0].shape[0] * len(prediction)
                    shape = tuple(shape)
                    preds = torch.zeros(shape).cuda()
                    bs_singlegpu = prediction[0].shape[0]
                    for i in range(len(prediction)):
                        preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += gaussian_importance_map
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def save_nii(args, pred, label, name, affine):  # B, c, WHD
    seg_pred_2class = np.asarray(np.around(pred), dtype=np.uint8)
    save_path = args.save_path
    seg_pred_0 = seg_pred_2class[:, 0, :, :, :]
    seg_pred_1 = seg_pred_2class[:, 1, :, :, :]
    seg_pred = np.zeros_like(seg_pred_0)
    if name[0][0:3] != 'spl':
        seg_pred = np.where(seg_pred_0 == 1, 1, seg_pred)
        seg_pred = np.where(seg_pred_1 == 1, 2, seg_pred)
    else:  # spleen only organ
        seg_pred = seg_pred_0

    label_0 = label[:, 0, :, :, :]
    label_1 = label[:, 1, :, :, :]
    seg_label = np.zeros_like(label_0)
    seg_label = np.where(label_0 == 1, 1, seg_label)
    seg_label = np.where(label_1 == 1, 2, seg_label)

    if name[0][0:3] != 'cas':
        seg_pred = seg_pred.transpose((0, 2, 3, 1))
        seg_label = seg_label.transpose((0, 2, 3, 1))

    # save
    for tt in range(seg_pred.shape[0]):
        seg_pred_tt = seg_pred[tt]
        seg_label_tt = seg_label[tt]
        seg_pred_tt = nib.Nifti1Image(seg_pred_tt, affine=affine[tt])
        seg_label_tt = nib.Nifti1Image(seg_label_tt, affine=affine[tt])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        seg_label_save_p = os.path.join(save_path + '/%s_label.nii.gz' % (name[tt]))
        seg_pred_save_p = os.path.join(save_path + '/%s_pred.nii.gz' % (name[tt]))
        nib.save(seg_label_tt, seg_label_save_p)
        nib.save(seg_pred_tt, seg_pred_save_p)
    return None


def validate(args, input_size, model, ValLoader, num_classes, engine):
    val_loss = torch.zeros(size=(7, 1)).cuda()  # np.zeros(shape=(7, 1))
    val_Dice = torch.zeros(size=(7, 2)).cuda()  # np.zeros(shape=(7, 2))
    count = torch.zeros(size=(7, 2)).cuda()  # np.zeros(shape=(7, 2))

    for index, batch in enumerate(ValLoader):

        image, label, name, task_id, affine = batch

        with torch.no_grad():

            pred_sigmoid = predict_sliding(args, model, image.numpy(), input_size, num_classes, task_id)

            loss = torch.tensor(1).cuda()
            val_loss[task_id[0], 0] += loss

            if label[0, 0, 0, 0, 0] == -1:
                dice_c1 = torch.from_numpy(np.array([-999]))
            else:
                dice_c1 = dice_score(pred_sigmoid[:, 0, :, :, :], label[:, 0, :, :, :])
                val_Dice[task_id[0], 0] += dice_c1
                count[task_id[0], 0] += 1
            if label[0, 1, 0, 0, 0] == -1:
                dice_c2 = torch.from_numpy(np.array([-999]))
            else:
                dice_c2 = dice_score(pred_sigmoid[:, 1, :, :, :], label[:, 1, :, :, :])
                val_Dice[task_id[0], 1] += dice_c2
                count[task_id[0], 1] += 1

            print('Task%d-%s loss:%.4f Organ:%.4f Tumor:%.4f' % (
            task_id, name, loss.item(), dice_c1.item(), dice_c2.item()))

            # save
            save_nii(args, pred_sigmoid, label, name, affine)

    count[count == 0] = 1
    val_Dice = val_Dice / count
    val_loss = val_loss / count.max(axis=1)[0].unsqueeze(1)

    reduce_val_loss = torch.zeros_like(val_loss).cuda()
    reduce_val_Dice = torch.zeros_like(val_Dice).cuda()
    for i in range(val_loss.shape[0]):
        reduce_val_loss[i] = engine.all_reduce_tensor(val_loss[i])
        reduce_val_Dice[i] = engine.all_reduce_tensor(val_Dice[i])

    if args.local_rank == 0:
        print("Sum results")
        for t in range(7):
            print('Sum: Task%d- loss:%.4f Organ:%.4f Tumor:%.4f' % (t, val_loss[t, 0], val_Dice[t, 0], val_Dice[t, 1]))

    return reduce_val_loss.mean(), reduce_val_Dice


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = 1234
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create network.
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

        model = nn.DataParallel(model)

        model.eval()

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')), False)
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        valloader, val_sampler = engine.get_test_loader(
            MOTSValDataSet(args.data_dir, args.val_list))

        # print('validate ...')
        val_loss, val_Dice = validate(args, input_size, [model], valloader, args.output_channel, engine)

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()

