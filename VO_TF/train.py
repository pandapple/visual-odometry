import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import random_split
import argparse

from voTransformer.model import VOTransformer

import matplotlib.pyplot as plt

def compute_loss(y_hat, y, criterion, weighted_loss, seq_len):
    if weighted_loss == None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], seq_len-1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], seq_len-1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        # compute custom loss
        k = weighted_loss
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss = loss_angles + loss_translation
    return loss

def compute_loss_R(y_hat, y, criterion, weighted_loss, seq_len):
    if weighted_loss == None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], seq_len-1, 12))
        gt_rotation = y[:, :, :9].flatten()
        gt_translation = y[:, :, 9:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], seq_len-1, 12))
        estimated_angles = y_hat[:, :, :9].flatten()
        estimated_translation = y_hat[:, :, 9:].flatten()

        # compute custom loss
        k = weighted_loss
        loss_angles = k * criterion(estimated_angles, gt_rotation.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss = loss_angles + loss_translation
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--weighted_loss', type=float, default=1.0)
    parser.add_argument('--seq_len', type=int, default=31)
    parser.add_argument('--epoch', type=int, default=1000)

    args = parser.parse_args()
    device = args.device

    train_loader = torch.load('processed_data/train.pth')
    eval_loader = torch.load('processed_data/eval.pth')

    model = VOTransformer(image_num=args.seq_len)
    model = model.to(device)
    # model.load_state_dict(torch.load('best_model.pth'))
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)

    best_loss = 100
    train_losses = []
    eval_losses = []

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
    ax.grid()
    ax.set_xlabel('epochs')
    ax.set_ylabel('acc & loss')

    for i in range(args.epoch):
        model.train()
        train_loss = 0
        train_num = 0
        for img_seq, pose_seq in train_loader:

            img_seq = img_seq.to(device)
            pose_seq = pose_seq.to(device)
            output = model(img_seq)

            loss = compute_loss(output, pose_seq, criterion, args.weighted_loss, args.seq_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_num = train_num + 1

        train_loss = train_loss / train_num

        print('epoch:', i)
        print('train loss:', train_loss.item())

        train_losses.append(train_loss.item())

        model.eval()

        eval_loss = 0
        eval_num = 0
        with torch.no_grad():
            for img_seq, pose_seq in eval_loader:
                img_seq = img_seq.to(device)
                pose_seq = pose_seq.to(device)
                output = model(img_seq)

                loss = compute_loss(output, pose_seq, criterion, args.weighted_loss, args.seq_len)

                eval_loss += loss
                eval_num = eval_num + 1

            eval_loss = eval_loss / eval_num
            print('eval loss:', eval_loss.item())


            eval_losses.append(eval_loss.item())

        x_ = list(range(i+1))
        ax.plot(x_, train_losses, color='blue', label='train loss')
        ax.plot(x_, eval_losses, color='red', label='eval loss')
        ax.legend()
        plt.draw()
        plt.pause(0.1)
        plt.savefig('1.png')

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), 'best_model.pth')
