import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from torch.utils.data import random_split

from voTransformer.superpoint import SuperPoint
from voTransformer.utils import rotation_to_euler

default_config = {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
            'fill_with_random_keypoints': False  # data augmentation for training the matcher
        }

model_pth = str(Path(__file__).parent) + "/superpoint_v1.pth"

device = 'cuda'

class VOdataloader(Dataset):
    def __init__(self, img_dir_list, pose_txt_list, seq_len):
        print("----------Start Data Loading----------")
        self.seq_len = seq_len
        self.img_dir_list = img_dir_list
        self.pose_txt_list = pose_txt_list

        self.superpoint = SuperPoint(default_config).to(device)
        self.superpoint.load_state_dict(torch.load(model_pth))

        self.poseSeqs = self.poseReader()
        self.imgSeqs = self.imgFeatReader()

        print("----------Finish Loading----------")

    def __len__(self):
        return self.imgSeqs.size(0)

    def __getitem__(self, idx):
        return self.imgSeqs[idx], self.poseSeqs[idx]

    def imgFeatReader(self):
        seqs_all = []
        for l in range(len(self.img_dir_list)):
            print(l)
            name_list = os.listdir(self.img_dir_list[l])
            name_list = sorted(name_list)

            seq_list = []
            for i in range(0, len(name_list), self.seq_len-1):
                framed_seq = []
                if i+self.seq_len >= len(name_list):
                    break
                for j in range(self.seq_len):
                    with torch.no_grad():
                        path = self.img_dir_list[l] + '/' + name_list[i+j]
                        # print(path)
                        img = cv2.imread(path)
                        height, width, channels = img.shape
                        img_raw = img.copy()
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img_gray = img.copy()
                        img = torch.from_numpy(img / 255.).float().unsqueeze(0).unsqueeze(0)
                        img = img.to(device)
                        pred = self.superpoint(img)
                        kpts = pred["keypoints"][0]
                        data_tensor_list = []
                        for k in range(kpts.size(0)):
                            v = int(kpts[k][0].item())
                            u = int(kpts[k][1].item())
                            data_list = []
                            data_list.append(u/376.)
                            data_list.append(v/1241.)
                            # (-3, -3) -- (3, 3)
                            for id1 in range(-3, 4):
                                for id2 in range(-3, 4):
                                    # data_list.append(img_raw[u + id1][v + id2][0]/255.)
                                    # data_list.append(img_raw[u + id1][v + id2][1]/255.)
                                    # data_list.append(img_raw[u + id1][v + id2][2]/255.)
                                    data_list.append(img_gray[u + id1][v + id2] / 255.)
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
                seq_list.append(framed_seq)

            seqs = torch.concat(seq_list)
            seqs_all.append(seqs)

        seqs_all = torch.concat(seqs_all)
        print(seqs_all.shape)

        return seqs_all

    def poseReader(self):
        seqs_all = []
        for l in range(len(self.pose_txt_list)):
            print(l)
            infile = open(self.pose_txt_list[l])
            lines = [line.split('\n')[0] for line in infile.readlines()]
            poses = [[float(value) for value in l.split(' ')] for l in lines]
            seq_list = []
            for i in range(0, len(poses), self.seq_len-1):
                if i+self.seq_len >= len(poses):
                    break

                vector_ref = np.array(poses[i])
                matrix_3x4_ref = vector_ref.reshape((3, 4))
                matrix_R_ref = matrix_3x4_ref[:, :3]
                matrix_t_ref = matrix_3x4_ref[:, 3:]
                inverse_matrix_R_ref = np.linalg.inv(matrix_R_ref)

                pose_list = []
                for j in range(1, self.seq_len):
                    vector_n = np.array(poses[i+j])
                    matrix_3x4_n = vector_n.reshape((3, 4))
                    matrix_R_n = matrix_3x4_n[:, :3]
                    matrix_t_n = matrix_3x4_n[:, 3:]
                    t = inverse_matrix_R_ref @ (matrix_t_n - matrix_t_ref)
                    R = inverse_matrix_R_ref @ matrix_R_n

                    euler_np = rotation_to_euler(R)
                    # print(np.linalg.norm(euler_np))

                    pose = torch.tensor([euler_np[0], euler_np[1], euler_np[2], t[0][0], t[1][0], t[2][0]])
                    pose = pose.unsqueeze(0)

                    pose_list.append(pose)

                pose_seq = torch.concat(pose_list)
                pose_seq = pose_seq.unsqueeze(0)
                seq_list.append(pose_seq)

            seqs = torch.concat(seq_list)
            seqs_all.append(seqs)

        seqs_all = torch.concat(seqs_all)
        print(seqs_all.shape)
        return seqs_all

if __name__ == '__main__':
    imgSeq_lists = ['/home/pan/dataset/00/image_2', '/home/pan/dataset/02/image_2',
                    '/home/pan/dataset/08/image_2', '/home/pan/dataset/09/image_2']
    poseSeq_lists = ['/home/pan/dataset/00/00.txt', '/home/pan/dataset/02/02.txt',
                     '/home/pan/dataset/08/08.txt', '/home/pan/dataset/09/09.txt']
    data = VOdataloader(imgSeq_lists, poseSeq_lists, 31)

    train_data, eval_data = random_split(data, [round(0.8 * data.__len__()), round(0.2 * data.__len__())],
                                         generator=torch.Generator().manual_seed(128))

    train_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    eval_loader = Data.DataLoader(dataset=eval_data, batch_size=1, shuffle=True)
    train_saved_data = []
    for img_seq, pose_seq in train_loader:
        train_saved_data.append((img_seq, pose_seq))
    train_save_tensor = [(data.cpu(), target.cpu()) for data, target in train_saved_data]
    torch.save(train_save_tensor, 'train.pth')

    eval_saved_data = []
    for img_seq, pose_seq in eval_loader:
        eval_saved_data.append((img_seq, pose_seq))
    eval_save_tensor = [(data.cpu(), target.cpu()) for data, target in eval_saved_data]
    torch.save(eval_save_tensor, 'eval.pth')
