import os
import argparse
import itertools
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F
import json

from crossLayerModel import crossMuEmbedNet
from data import DBP15K
from loss import L1_Loss
from utils import add_inverse_rels, get_train_batch, get_hits, get_hits_stable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="./data/DBP15K")

    # parser.add_argument("--data", default="./EA_Data")
    # 修改第一个位置
    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--output_dir", default="./crossOutput")

    parser.add_argument("--rate", type=float, default=0.85)
    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=3)

    parser.add_argument("--epoch", type=int, default=340)
    parser.add_argument("--neg_epoch", type=int, default=5)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--stable_test", action="store_true", default=True)
    args = parser.parse_args()
    return args


def init_data(args, device):
    data = DBP15K(args.data, args.lang, rate=args.rate)[0]
    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)
    return data


def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    return x1, x2
def test(model, data, criterion, args, stable=False):
    x1, x2 = get_emb(model, data)
    print('-' * 16 + 'Train_set' + '-' * 16)
    get_hits(x1, x2, data.train_set)

    print('-' * 16 + 'Test_set' + '-' * 17)
    get_hits(x1, x2, data.test_set)

    # 生成测试集的负样本
    test_batch = get_train_batch(x1, x2, data.test_set, args.k)

    # 使用与训练集相同的方法计算测试集的损失
    test_loss = criterion(x1, x2, data.test_set, test_batch)
    print('Test Loss: %.3f' % test_loss)

    if stable:
        get_hits_stable(x1, x2, data.test_set)
    print()
    return x1, x2



def train(model, criterion, optimizer, data, train_batch):
    model.train()
    x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
    x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    loss = criterion(x1, x2, data.train_set, train_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)
    model = crossMuEmbedNet(data.x1.size(1), args.r_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = L1_Loss(args.gamma)

    # 定义列表保存训练和测试损失
    train_losses = []
    test_losses = []

    # 创建日志文件
    # 修改第二个位置
    log_file_path = os.path.join(args.output_dir, 'cross_zh_en_log1.txt')

    with open(log_file_path, 'w') as log_file:
        for epoch in range(args.epoch):
            if epoch % args.neg_epoch == 0:
                x1, x2 = get_emb(model, data)
                train_batch = get_train_batch(x1, x2, data.train_set, args.k)

            # 训练模型并计算损失
            loss = train(model, criterion, optimizer, data, train_batch)
            if(epoch+1)%5==0:
                train_losses.append(loss.item())
                log = f'Epoch: {epoch + 1} / {args.epoch} \tLoss: {loss:.3f}'
                print(log)
                log_file.write(log + '\n')

            # 定期评估模型性能
            if (epoch + 1) % args.test_epoch == 0:
                print()
                x1, x2 = get_emb(model, data)
                test_batch = get_train_batch(x1, x2, data.test_set, args.k)
                test_loss = criterion(x1, x2, data.test_set, test_batch)
                if (epoch + 1) % 5 == 0:
                    test_losses.append(test_loss.item())
                    log = f'Test Loss: {test_loss:.3f}'
                    # print(log)
                    log_file.write(log + '\n')
                test(model, data, criterion, args, args.stable_test)

    x1, x2 = get_emb(model, data)
    # 修改第三个位置
    model_sava_path = os.path.join(args.output_dir, 'cross_zh_en1.pt')
    # model_sava_path = os.path.join(args.output_dir, 'A3_zh_en.pt')
    torch.save([x1[data.test_set[:, 0]].cpu(), x2[data.test_set[:, 1]].cpu()], model_sava_path)

    # 保存训练和测试损失到文件
    # 修改第四个位置
    with open('crossOutput/cross_zh_en_losses1.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'test_losses': test_losses}, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)