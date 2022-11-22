import os

import numpy as np

import nibabel as nib
from skimage.measure import label as LAB

import argparse

from medpy.metric import hd95


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CQTr post processing!")

    parser.add_argument("--img_folder_path", type=str, default='outputs/')  # outputs/dodnet/
    parser.add_argument("--postp_outputs", type=str, default='postp_outputs/')  # outputs/dodnet/
    return parser.parse_args()


args = get_arguments()


def continues_region_extract_organ(label, keep_region_nums):  # keep_region_nums=1
    mask = False * np.zeros_like(label)
    regions = np.where(label >= 1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

    #
    ary_num = np.zeros(shape=(n + 1, 1))
    for i in range(0, n + 1):
        ary_num[i] = np.sum(L == i)
    max_index = np.argsort(-ary_num, axis=0)
    count = 1
    for i in range(1, n + 1):
        if count <= keep_region_nums:  # keep
            mask = np.where(L == max_index[i][0], label, mask)
            count += 1
    label = np.where(mask == True, label, np.zeros_like(label))
    return label


def continues_region_extract_tumor(label):  #

    regions = np.where(label >= 1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

    for i in range(1, n + 1):
        if np.sum(L == i) <= 50 and n > 1:  # remove default 50
            label = np.where(L == i, np.zeros_like(label), label)

    return label


def locate_judge(pred, dis=200):
    pred_organ = (pred == 1)
    pred_tumor = (pred == 2)
    #
    L, n = LAB(pred_tumor, neighbors=4, background=0, connectivity=2, return_num=True)
    if n > 0:
        ary_num = np.zeros(shape=(n + 1, 1))
        for i in range(0, n + 1):
            ary_num[i] = np.sum(L == i)
        max_index = np.argsort(-ary_num, axis=0)

        label = pred
        location_organ = np.where(label == 1)
        location_tumor = np.where(L == max_index[1])
        deltax = np.mean(location_organ[0]) - np.mean(location_tumor[0])
        deltay = np.mean(location_organ[1]) - np.mean(location_tumor[1])
        deltaz = np.mean(location_organ[2]) - np.mean(location_tumor[2])
        if np.sqrt(deltax ** 2 + deltay ** 2 + deltaz ** 2) > dis:
            pred_tumor = np.where(L == max_index[1], np.zeros_like(pred_tumor), pred_tumor)
        return pred_tumor * 2 + pred_organ
    else:
        return pred


def distance(maskA, maskB):
    locatesA = np.array(np.where(maskA != 0))
    locatesB = np.array(np.where(maskB != 0))
    return np.linalg.norm(np.mean(locatesA, axis=1) - np.mean(locatesB, axis=1))


def locate_judgeh(pred_organ, tumor, label_tumor, name, dis=215):
    L, n = LAB(tumor, neighbors=4, background=0, connectivity=2, return_num=True)
    pred_tumor = np.zeros_like(tumor)
    if n > 0:
        ary_num = np.zeros(shape=(n + 1, 1))
        for i in range(0, n + 1):
            ary_num[i] = np.sum(L == i)
        max_index = np.argsort(-ary_num, axis=0)
        for i in range(1, n + 1):
            d = distance(pred_organ, L == max_index[i])
            if d < dis:
                # print(name,d)
                pred_tumor = pred_tumor + np.where(L == max_index[i], tumor, np.zeros_like(tumor))
        return pred_tumor


def expand(pred_tumor, pred_organ):
    L, n = LAB(pred_organ, neighbors=4, background=0, connectivity=2, return_num=True)
    ary_num = np.zeros(shape=(n + 1, 1))
    for i in range(0, n + 1):
        ary_num[i] = np.sum(L == i)
    max_index = np.argsort(-ary_num, axis=0)
    if max_index.shape[0] > 7:
        N = 7
    else:
        N = max_index.shape[0]
    for i in range(N):
        if i == 0:
            continue
        if distance(L == max_index[i], pred_tumor) < 30:
            pred_tumor = pred_tumor + (L == max_index[i]).astype(np.int8)
    return pred_tumor


def dice_score(preds, labels):
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) + 1

    dice = 2 * num / den

    return dice.mean()


def task_index(name):
    if "liver" in name:
        return 0
    if "case" in name:
        return 1
    if "hepa" in name:
        return 2
    if "pancreas" in name:
        return 3
    if "colon" in name:
        return 4
    if "lung" in name:
        return 5
    if "spleen" in name:
        return 6


def compute_HD95(ref, pred):
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))


def save_nii(args, pred, name, affine):  # B, c, WHD
    seg_pred_2class = np.asarray(np.around(pred), dtype=np.uint8)
    seg_pred_0 = seg_pred_2class[0, :, :, :]
    seg_pred_1 = seg_pred_2class[1, :, :, :]
    seg_pred = np.zeros_like(seg_pred_0)
    if name[0:3] != 'spl':
        seg_pred = np.where(seg_pred_0 == 1, 1, seg_pred)
        seg_pred = np.where(seg_pred_1 == 1, 2, seg_pred)
    else:  # spleen only organ
        seg_pred = seg_pred_0

    # save
    seg_pred_NII = nib.Nifti1Image(seg_pred, affine=affine)
    if not os.path.exists(args.postp_outputs):
        os.makedirs(args.postp_outputs)
    seg_pred_save_p = os.path.join(args.postp_outputs + name.replace('pred', 'postp'))
    nib.save(seg_pred_NII, seg_pred_save_p)
    return None


val_Dice = np.zeros(shape=(7, 2))
val_HD = np.zeros(shape=(7, 2))
count = np.zeros(shape=(7, 2))

for root, dirs, files in os.walk(args.img_folder_path):
    for i in sorted(files):
        if i[-12:-7] != 'label':
            continue
        i2 = i[:-12] + 'pred' + i[-7:]
        i_file = root + i
        i2_file = root + i2
        predNII = nib.load(i2_file)
        labelNII = nib.load(i_file)
        pred = predNII.get_data()
        label = labelNII.get_data()
        affine = labelNII.affine

        # post-processing

        task_id = task_index(i)
        if task_id == 0 or task_id == 1 or task_id == 3:
            if task_id == 1:
                pred = locate_judge(pred)
            pred_organ = (pred >= 1)
            pred_tumor = (pred == 2)
            label_organ = (label >= 1)
            label_tumor = (label == 2)
        elif task_id == 2:
            pred_organ = (pred == 1)
            pred_tumor = (pred == 2)
            label_organ = (label == 1)
            label_tumor = (label == 2)

        elif task_id == 4:
            pred_organ = None
            pred_tumor = (pred == 2)
            label_organ = None
            label_tumor = (label == 2)
        elif task_id == 5:
            pred_organ = (pred == 1)
            pred_tumor = (pred == 2)
            label_organ = None
            label_tumor = (label == 2)
        elif task_id == 6:
            pred_organ = (pred == 1)
            pred_tumor = None
            label_organ = (label == 1)
            label_tumor = None
        else:
            print("No such a task!")

        if task_id == 0:
            pred_organ = continues_region_extract_organ(pred_organ, 1)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continues_region_extract_tumor(pred_tumor)
            save_nii(args,np.stack((pred_organ,pred_tumor),axis=0),i2,affine)
        elif task_id == 1:
            pred_organ = continues_region_extract_organ(pred_organ, 2)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continues_region_extract_organ(pred_tumor, 1)
            save_nii(args,np.stack((pred_organ,pred_tumor),axis=0),i2,affine)
        elif task_id == 2:
            temp = continues_region_extract_organ(pred_tumor, 3)
            tumor = continues_region_extract_tumor(temp)
            pred_tumor = locate_judgeh(pred_organ, tumor, label_tumor, i, 138)
            save_nii(args,np.stack((pred_organ,pred_tumor),axis=0),i2,affine)
        elif task_id == 3:
            pred_organ = continues_region_extract_organ(pred_organ, 1)
            pred_tumor = np.where(pred_organ == True, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continues_region_extract_tumor(pred_tumor)
            save_nii(args,np.stack((pred_organ,pred_tumor),axis=0),i2,affine)
        elif task_id == 4:
            pred_tumor = continues_region_extract_organ(pred_tumor, 1)
            save_nii(args,np.stack((np.zeros_like(pred_tumor),pred_tumor),axis=0),i2,affine)
        elif task_id == 5:
            pred_tumor = continues_region_extract_organ(pred_tumor, 1)
            pred_tumor = expand(pred_tumor, pred_organ)
            save_nii(args,np.stack((np.zeros_like(pred_tumor),pred_tumor),axis=0),i2,affine)
        elif task_id == 6:
            pred_organ = continues_region_extract_organ(pred_organ, 1)
            save_nii(args,np.stack((pred_organ,np.zeros_like(pred_organ)),axis=0),i2,affine)
        else:
            print("No such a task index!!!")

        if label_organ is not None:
            dice_c1 = dice_score(pred_organ, label_organ)
            HD_c1 = compute_HD95(label_organ, pred_organ)
            val_Dice[task_id, 0] += dice_c1
            val_HD[task_id, 0] += HD_c1
            count[task_id, 0] += 1
        else:
            dice_c1 = -999
            HD_c1 = 999
        if label_tumor is not None:

            dice_c2 = dice_score(pred_tumor, label_tumor)
            HD_c2 = compute_HD95(label_tumor, pred_tumor)
            val_Dice[task_id, 1] += dice_c2
            val_HD[task_id, 1] += HD_c2
            count[task_id, 1] += 1
        else:
            dice_c2 = -999
            HD_c2 = 999
        print("%s: Organ_Dice %f, tumor_Dice %f | Organ_HD %f, tumor_HD %f" % (i[:-13], dice_c1, dice_c2, HD_c1, HD_c2))

count[count == 0] = 1
val_Dice = val_Dice / count
val_HD = val_HD / count

print("Sum results")
for t in range(7):
    print('Sum: Task%d- Organ_Dice:%.4f Tumor_Dice:%.4f | Organ_HD:%.4f Tumor_HD:%.4f' % (
    t, val_Dice[t, 0], val_Dice[t, 1], val_HD[t, 0], val_HD[t, 1]))
