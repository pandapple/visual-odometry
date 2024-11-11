import os
import cv2
import numpy as np
from pathlib import Path
from voTransformer.utils import rotation_to_euler

if __name__ == '__main__':
    seq_len = 31
    pose_txt_list = ['/home/pan/dataset/00/00.txt', '/home/pan/dataset/01/01.txt', '/home/pan/dataset/02/02.txt', '/home/pan/dataset/03/03.txt',
                     '/home/pan/dataset/04/04.txt', '/home/pan/dataset/05/05.txt', '/home/pan/dataset/06/06.txt', '/home/pan/dataset/07/07.txt',
                     '/home/pan/dataset/08/08.txt', '/home/pan/dataset/09/09.txt']
    overlap = round((seq_len - 1) / 2)
    for l in range(len(pose_txt_list)):
        print(l)
        txt_name = str(l) + '.txt'
        file = open(txt_name, mode='w')
        infile = open(pose_txt_list[l])
        lines = [line.split('\n')[0] for line in infile.readlines()]
        poses = [[float(value) for value in l.split(' ')] for l in lines]
        for i in range(0, len(poses), overlap):
            if i + seq_len >= len(poses):
                break

            vector_ref = np.array(poses[i])
            matrix_3x4_ref = vector_ref.reshape((3, 4))
            matrix_R_ref = matrix_3x4_ref[:, :3]
            matrix_t_ref = matrix_3x4_ref[:, 3:]
            inverse_matrix_R_ref = np.linalg.inv(matrix_R_ref)

            for j in range(1, seq_len):
                vector_n = np.array(poses[i + j])
                matrix_3x4_n = vector_n.reshape((3, 4))
                matrix_R_n = matrix_3x4_n[:, :3]
                matrix_t_n = matrix_3x4_n[:, 3:]
                t = inverse_matrix_R_ref @ (matrix_t_n - matrix_t_ref)
                R = inverse_matrix_R_ref @ matrix_R_n

                euler_np = rotation_to_euler(R)

                line = [str(euler_np[0]), ' ', str(euler_np[1]), ' ', str(euler_np[2]), ' ', str(t[0][0]), ' ', str(t[1][0]), ' ', str(t[2][0]), '\n']
                file.writelines(line)

        file.close()