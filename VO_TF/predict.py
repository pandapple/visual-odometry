import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from torch.utils.data import random_split

import time
import math

from voTransformer.model3 import VOTransformer
# from voTransformer.model_separate import VOTransformer_rotation
# from voTransformer.model2 import VOTransformer
from voTransformer.superpoint import SuperPoint
from voTransformer.utils import rotation_to_euler

from voTransformer.utils import euler_to_rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_loss(y_hat, y, criterion, weighted_loss, seq_len):
    if weighted_loss == None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], seq_len-1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        y_hat = torch.reshape(y_hat, (y_hat.shape[0], seq_len-1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        k = weighted_loss
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss = loss_angles + loss_translation
    return loss

def predict_one(model, img_seq, pose_gth, criterion, weighted_loss):
    pose_pred = model(img_seq)
    loss = compute_loss(pose_pred, pose_gth, criterion, weighted_loss, img_seq.shape[1])
    print('predicted loss:', loss.item())

    return loss.item(), pose_pred

def seq_predict_pose(vo_model, sp_model, img_dir, seq_len):
    name_list = os.listdir(img_dir)
    name_list = sorted(name_list)

    pose_list = []
    pose_list.append(np.array([0.0, 0.0, 0.0]))
    matrix_R_ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    matrix_t_ref = np.array([[0.0], [0.0], [0.0]])

    overlap = round((seq_len-1)/2.0)
    for i in range(0, len(name_list), overlap):
        framed_seq = []
        if i + seq_len >= len(name_list):
            break
        for j in range(seq_len):
            path = img_dir + '/' + name_list[i + j]
            img = cv2.imread(path)
            # height, width, channels = img.shape
            img_raw = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = torch.from_numpy(img / 255.).float().unsqueeze(0).unsqueeze(0)
            img = img.to(device)
            pred = sp_model(img)
            kpts = pred["keypoints"][0]

            datalist = []
            for k in range(kpts.size(0)):
                if k >= 256:
                    break
                v = int(kpts[k][0].item())
                u = int(kpts[k][1].item())
                r = img_raw[u, v, 0]
                g = img_raw[u, v, 1]
                b = img_raw[u, v, 2]

                datalist.append(u)
                datalist.append(v)
                datalist.append(r)
                datalist.append(g)
                datalist.append(b)

            framed = torch.tensor(datalist)
            framed = framed.to(torch.float32)
            framed = framed.unsqueeze(dim=0)
            framed_seq.append(framed)

        framed_seq = torch.concat(framed_seq, dim=0)
        framed_seq = framed_seq.unsqueeze(0)

        framed_seq = framed_seq.to(device)
        pose_pred = vo_model(framed_seq)
        pose_pred = pose_pred[0]
        # print(pose_pred)

        for p in range(overlap):
            R_to_ref = euler_to_rotation(pose_pred[p, 0].item(), pose_pred[p, 1].item(), pose_pred[p, 2].item())
            t_to_ref = np.array([[pose_pred[p, 3].item()], [pose_pred[p, 4].item()], [pose_pred[p, 5].item()]])

            R_to_1 = matrix_R_ref @ R_to_ref
            t_to_1 = matrix_R_ref @ t_to_ref + matrix_t_ref

            t_global_ = np.array([t_to_1[0][0], t_to_1[1][0], t_to_1[2][0]])

            print(t_global_)

            if p == overlap - 1:
                matrix_R_ref = R_to_1
                matrix_t_ref = t_to_1

            pose_list.append(t_global_)
    return pose_list

def seq_predict_pose(vo_model, sp_model, img_dir):
    name_list = os.listdir(img_dir)
    name_list = sorted(name_list)

    pose_list = []
    pose_list.append(np.array([0.0, 0.0, 0.0]))
    matrix_R_ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    matrix_t_ref = np.array([[0.0], [0.0], [0.0]])

    T1 = time.time()
    time_list = []
    for i in range(0, len(name_list), 30):
        T3 = time.time()
        framed_seq = []
        if i + 31 >= len(name_list):
            break
        # T1 = time.time()
        for j in range(31):
            with torch.no_grad():
                path = img_dir + '/' + name_list[i + j]
                img = cv2.imread(path)
                height, width, channels = img.shape
                img_raw = img.copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = torch.from_numpy(img / 255.).float().unsqueeze(0).unsqueeze(0)
                img = img.to(device)
                pred = sp_model(img)
                kpts = pred["keypoints"][0]
                data_tensor_list = []
                for k in range(kpts.size(0)):
                    v = int(kpts[k][0].item())
                    u = int(kpts[k][1].item())
                    data_list = []
                    data_list.append(u)
                    data_list.append(v)
                    # (-1, -1) -- (1, 1)
                    for id1 in range(-1, 2):
                        for id2 in range(-1, 2):
                            data_list.append(img_raw[u + id1][v + id2][0])
                            data_list.append(img_raw[u + id1][v + id2][1])
                            data_list.append(img_raw[u + id1][v + id2][2])
                    data_array = np.array(data_list)
                    data_tensor = torch.tensor(data_array)
                    data_tensor = data_tensor.unsqueeze(0)
                    data_tensor_list.append(data_tensor)

                framed = torch.concat(data_tensor_list)
                framed = framed[:128, :]
                framed = framed.to(torch.float32)
                framed = framed.unsqueeze(dim=0)
                framed_seq.append(framed)

        framed_seq = torch.concat(framed_seq, dim=0)
        framed_seq = framed_seq.unsqueeze(0)
        framed_seq = framed_seq.to(device)
        pose_pred = vo_model(framed_seq)
        pose_pred = pose_pred[0]

        for p in range(pose_pred.size(0)):
            R_to_ref = euler_to_rotation(pose_pred[p, 0].item(), pose_pred[p, 1].item(), pose_pred[p, 2].item())
            t_to_ref = np.array([[pose_pred[p, 3].item()], [pose_pred[p, 4].item()], [pose_pred[p, 5].item()]])

            R_to_1 = matrix_R_ref @ R_to_ref
            t_to_1 = matrix_R_ref @ t_to_ref + matrix_t_ref

            t_global_ = np.array([t_to_1[0][0], t_to_1[1][0], t_to_1[2][0]])

            print(t_global_)

            if p == pose_pred.size(0) - 1:
                matrix_R_ref = R_to_1
                matrix_t_ref = t_to_1

            pose_list.append(t_global_)
        T4 = time.time()
        ave_1 = (T4 - T3) * 1000 / 15
        time_list.append(ave_1)
    T2 = time.time()
    ave = (T2 - T1) * 1000 / len(pose_list)
    std = np.std(time_list)
    print('ave time:', ave)
    print('std:', std)
    return pose_list

if __name__ == '__main__':

    device = 'cuda'

    model = VOTransformer(image_num=31)
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    model.eval()

    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'fill_with_random_keypoints': False  # data augmentation for training the matcher
    }
    superpoint = SuperPoint(default_config).to(device)
    superpoint.load_state_dict(torch.load('voTransformer/superpoint_v1.pth'))
    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)
    pose_list = seq_predict_pose(model, superpoint, '/home/pan/dataset/01/image_2')

    datax = []
    datay = []
    dataz = []

    for i in range(len(pose_list)):
        datax.append(pose_list[i][0])
        datay.append(pose_list[i][1])
        dataz.append(pose_list[i][2])
       
    gt_x = []
    gt_y = []
    gt_z = []
    infile = open('/home/pan/dataset/01/01.txt')
    lines = [line.split('\n')[0] for line in infile.readlines()]
    poses = [[float(value) for value in l.split(' ')] for l in lines]
    for i in range(len(poses)):
        gt_x.append(poses[i][3])
        gt_y.append(poses[i][7])
        gt_z.append(poses[i][11])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(datax, datay, dataz, c='r')
    ax.plot(gt_x, gt_y, gt_z, c='b')

    plt.show()
    plt.savefig('traj.png')
