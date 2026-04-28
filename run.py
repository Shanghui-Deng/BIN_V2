from __future__ import print_function, division
import argparse
import time
from email.policy import default
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.optim import Adam
import itertools
from queue import Queue
from itertools import chain
import torch_clustering
from instance_loss import InstanceLoss
from models import PMVCR_2view
import os
from data_loader import load_data, data_process
import logging
from configure import get_default_config
from sklearn.decomposition import PCA
from utils import euclidean_dist_2v
import random
from Network import Network
from utils import shuffle_data, next_batch
import torch.nn.functional as F
from clusteringPerformance import clusteringMetrics
from Dataloader import MultiViewDatasetLoader
from torch.utils.data import DataLoader
from utils import CrossEn, KL


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Configure logging
def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def plot_TSNET(args, z, label, num_cluster, epoch):
    tsne = TSNE(n_components=2, random_state=10, learning_rate=200)
    # label = label.cpu().detach().numpy()
    z_embedded = tsne.fit_transform(z.cpu().detach().numpy())
    plt.figure()
    colors = ['#1f77b4', '#2ca02c', '#9467bd',
              '#e377c2', '#bcbd22',
              '#aec7e8', '#98df8a', '#c5b0d5',
              '#f7b6d2', '#dbdb8d',
              '#c7e9c0', '#bcbd22', '#8c6d31',
              '#e7ba52', '#ce6dbd',
              '#3182bd', '#31a354', '#636363',
              '#dadaeb', '#fccde5',  ###
              '#ff7f0e', '#d62728', '#8c564b', '#7f7f7f', '#17becf', '#ffbb78', '#ff9896', '#c49c94', '#c7c7c7',
              '#9edae5', '#fddaec', '#ad494a', '#6b4c9a', '#a55194', '#de9ed6', '#e6550d', '#756bb1', '#9e9ac8',
              '#bdbdbd', '#ff7f0e']
    for j in range(num_cluster):
        plt.scatter(z_embedded[label == j, 0], z_embedded[label == j, 1], s=3, color=colors[j % len(colors)])
    plt.axis('off')
    plt.savefig('/home/shanghui/dsh/storage/result/GHAP/align/{}/Visual/{}.pdf'.format(args.dataset, epoch),
                format='pdf', bbox_inches='tight', dpi=400)


def main():
    # Ensure environment variables and other setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_cuda = torch.cuda.is_available()
    logging.info(f"GPU available: {use_cuda}")

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(device)
    # best_scores_kmeans = [0, 0, 0, 0, 0, 0, 0, 0]
    best_epoch = 0

    args = parser.parse_args()

    # Set up logging file
    log_path = f"./result_log/{args.dataset}.log"
    setup_logging(log_path)

    args.cuda = torch.cuda.is_available()
    # logging.info(f"Using CUDA: {args.cuda}")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.dataset == 'Scene-15':
        args.lr_train = 0.0001
        args.seed = 2024
    elif args.dataset == 'RGB-D':
        args.seed = 24
        args.beta = 1
        args.alpha = 1
        args.lr_train = 1e-5
        args.main_view = 1
        args.con_epochs = 200
    elif args.dataset == 'DHA':
        args.lr_train = 0.000001
        # args.seed = 0
        # args.n_z = 128
        args.main_view = 0
        args.con_epochs = 500
        if args.unalign_ratio in [0.1,0.5,0.7]:
            args.seed = 42
        elif args.unalign_ratio in [0.3]:
            args.seed = 3
        elif args.unalign_ratio in [0.9]:
            args.seed = 6
        elif args.unalign_ratio in [1]:
            args.seed = 44
    elif args.dataset == 'Wiki_fea':
        args.lr_train = 0.00002
        # args.seed = 0
        args.main_view = 0
        args.con_epochs = 2000
        if args.unalign_ratio in [0.1]:
            args.seed = 8
        elif args.unalign_ratio in [0.3,0.5]:
            args.seed = 10
        elif args.unalign_ratio in [0.7]:
            args.seed = 3
        elif args.unalign_ratio in [0.9]:
            args.seed = 7
        elif args.unalign_ratio in [1]:
            args.seed = 42
    elif args.dataset == 'bbcsprot':
        # args.lr_train = 0.002
        # args.seed = 0
        args.main_view = 0
        args.con_epochs = 300
        if args.unalign_ratio in [0.1]:
            args.lr_train = 0.003
            args.seed = 0
        elif args.unalign_ratio in [0.3]:
            args.lr_train = 0.001
            args.seed = 0
        elif args.unalign_ratio in [0.5]:
            args.lr_train = 0.002
            args.seed = 0
        elif args.unalign_ratio in [0.7]:
            args.lr_train = 0.002
            args.seed = 9
        elif args.unalign_ratio in [0.9]:
            args.lr_train = 0.004
            args.seed = 9
        elif args.unalign_ratio in [1]:
            args.lr_train = 0.006
            args.seed = 0
    elif args.dataset == 'STL-10-2V':
        args.lr_train = 0.00001
        args.seed = 0
        args.main_view = 0
        args.con_epochs = 50
        args.beta = 1
        args.alpha = 1
    elif args.dataset == 'ALOI-100':
        args.seed = 0
        args.main_view = 0
        args.con_epochs = 50
        if args.unalign_ratio in [0.1,0.5]:
            args.lr_train = 3e-7
        elif args.unalign_ratio in [0.3]:
            args.lr_train = 2e-7
        elif args.unalign_ratio in [0.7]:
            args.lr_train = 5e-7
        elif args.unalign_ratio in [0.9]:
            args.lr_train = 4e-7
        elif args.unalign_ratio in [1]:
            args.lr_train = 1e-7
    elif args.dataset == 'Mfeat':
        args.lr_train = 0.0001
        args.seed = 0
        args.main_view = 0
    elif args.dataset == 'prokaryotic':
        # args.lr_train = 0.00001
        # args.seed = 0
        args.main_view = 0
    elif args.dataset == 'GSE100866':
        # args.lr_train = 0.00001
        args.seed = 0
        args.main_view = 0
        args.con_epochs = 300
        if args.unalign_ratio ==0.1 or args.unalign_ratio ==0.3:
            args.lr_train = 0.0006
        elif args.unalign_ratio ==0.5 or args.unalign_ratio ==0.9:
            args.lr_train = 0.0004
        elif args.unalign_ratio ==0.7:
            args.lr_train = 0.0003
        elif args.unalign_ratio ==1:
            args.lr_train = 0.0007
    elif args.dataset =='Cora':
        args.main_view =1
        args.seed=0
        args.lr_train = 0.001
        args.con_epochs = 500
    set_seed(args.seed)
    dataloader, num_sample, dim_list, aligned_index, unaligned_index, y = MultiViewDatasetLoader(args)
    train_loader = DataLoader(dataloader, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataloader, batch_size=num_sample, shuffle=False)
    args.num_cluster = np.size(np.unique(y[0]))
    print('num_cluster:', args.num_cluster)
    best_scores_kmeans = [0, 0, 0, 0, 0, 0, 0, 0]
    second_scores = [0, 0, 0, 0, 0, 0, 0, 0]
    third_scores = [0, 0, 0, 0, 0, 0, 0, 0]
    logging.info(
        'unalign_ratio:{} lr:{} seed:{} main_view:{} alpha:{} beta:{}'.format(args.unalign_ratio, args.lr_train,
                                                                              args.seed, args.main_view, args.alpha,
                                                                              args.beta))
    network = Network(args, dim_list, device).to(device)
    # print(network)
    optimizer = torch.optim.Adam(
        chain(network.parameters()),
        lr=args.lr_train,
        weight_decay=0.0000001
    )


    for epoch in range(args.con_epochs):
        loss_all = []
        rec_loss_all = []
        S_loss_all = []
        B_loss_all = []
        loss_fct = CrossEn()
        kl = KL()
        for batch_idx, (data0, data1, true_label) in enumerate(train_loader):
            data0 = data0.to(device)
            data1 = data1.to(device)
            re_data0, re_data1, z0, z1, M_t2v_logits, M_v2t_logits, logits, banzhaf, teacher = network(data0, data1)
            rec_loss = F.mse_loss(re_data0, data0) + F.mse_loss(re_data1, data1)
            rec_loss_all.append(rec_loss.item())
            S_loss_t2v = loss_fct(M_t2v_logits)
            S_loss_v2t = loss_fct(M_v2t_logits)
            S_loss = (S_loss_t2v + S_loss_v2t) / 2
            s_loss = kl(banzhaf, teacher) + kl(banzhaf.T, teacher.T)
            S_loss_all.append((S_loss).item())
            B_loss_all.append((s_loss).item())
            loss = rec_loss  +args.beta* S_loss + args.alpha*s_loss
            loss_all.append(loss.item())
            loss.backward()
            optimizer.step()
        print(
            f"Epoch {epoch} loss: {np.mean(loss_all)} Rec_loss: {np.mean(rec_loss_all)} S_loss: {np.mean(S_loss_all)} B_loss:{np.mean(B_loss_all)} ")
        network.eval()
        with (torch.no_grad()):
            for batch_idx, (data0, data1, true_label) in enumerate(test_loader):
                data0 = data0.to(device)
                data1 = data1.to(device)
                z0 = network.encoders[0](data0)
                z1 = network.encoders[1](data1)

                # h0 = network.mlps[0](z0)
                # h1 = network.mlps[1](z1)
                if args.unalign_ratio != 0:
                    sim_marix = torch.mm(z0[unaligned_index], z1[unaligned_index].t())
                    contrib0 = torch.norm(z0[unaligned_index], dim=1, keepdim=True)
                    contrib1 = torch.norm(z1[unaligned_index], dim=1, keepdim=True)
                    interaction = sim_marix - contrib0 - contrib1
                    # C = euclidean_dist_2v(z0[unaligned_index],z1[unaligned_index])
                    if args.main_view == 0:
                        data_reranged = data0.clone()
                        for i in range(len(unaligned_index)):
                            idx = torch.argsort(interaction[:, i],descending=True)
                            data_reranged[unaligned_index[i]] = data0[unaligned_index][idx[0]]
                        for batch_idx, (data0, data1, true_label) in enumerate(test_loader):
                            data1 = data1.to(device)
                            z0 = network.encoders[0](data_reranged)
                            z1 = network.encoders[1](data1)
                    elif args.main_view == 1:
                        data_reranged = data1.clone()
                        for i in range(len(unaligned_index)):
                            idx = torch.argsort(interaction[:, i], descending=True)
                            data_reranged[unaligned_index[i]] = data1[unaligned_index][idx[0]]
                        for batch_idx, (data0, data1, true_label) in enumerate(test_loader):
                            data0 = data0.to(device)
                            z0 = network.encoders[0](data0)
                            z1 = network.encoders[1](data_reranged)

                y1 = torch.tensor(true_label, dtype=torch.int).to(device).detach().cpu().numpy()
                z_both = torch.cat((z0, z1), dim=1)
                # z_both = torch.add(z0,z1)
                # z_both = torch.mean(torch.stack([z0,z1]),dim=0)
                kwargs = {
                    'metric': 'cosine',
                    'distributed': False,
                    'random_state': args.seed,
                    'n_clusters': args.num_cluster,
                    'verbose': False
                }
                km_torch = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
                psedo_labels = km_torch.fit_predict(z_both)
                ACC, NMI, ARI, Purity, Fscore, Precision, Recall, AMI = clusteringMetrics(y1,
                                                                                          psedo_labels.cpu().numpy())
                ACC = np.round(ACC, 4).item()
                NMI = np.round(NMI, 4).item()
                ARI = np.round(ARI, 4).item()
                Purity = np.round(Purity, 4).item()
                Fscore = np.round(Fscore, 4).item()
                Precision = np.round(Precision, 4).item()
                Recall = np.round(Recall, 4).item()
                AMI = np.round(AMI, 4).item()
                scores = [ACC, NMI, ARI, Purity, Fscore, Precision, Recall, AMI]
                ss = dict(
                    {'Epoch': epoch, 'ACC': ACC, 'NMI': NMI, 'ARI': ARI, 'Purity': Purity, 'F-score': Fscore,
                     'Precision': Precision,
                     'Recall': Recall, 'AMI': AMI})
                # logging.info(ss)
                print(ss)
                # plot_TSNET(args,z_both,psedo_labels.cpu().detach().numpy(),args.num_cluster,epoch)
                if scores[0] > best_scores_kmeans[0]:
                    best_scores_kmeans = scores
                    best_epoch = epoch
                # if scores[0] < best_scores_kmeans[0] and scores[0] > second_scores[0]:
                #     second_scores = scores
                # if scores[0] < best_scores_kmeans[0] and scores[0] < second_scores[0] and scores[0] > third_scores[0]:
                #     third_scores = scores
                    # np.save('/home/shanghui/dsh/storage/result/GHAP/align/{}/z0.npy'.format(args.dataset), z0.cpu().detach().numpy())
                    # np.save('/home/shanghui/dsh/storage/result/GHAP/align/{}/z1.npy'.format(args.dataset),
                    #         z0.cpu().detach().numpy())
                    # np.save('/home/shanghui/dsh/storage/result/GHAP/align/{}/z_all.npy'.format(args.dataset),
                    #         z_both.cpu().detach().numpy())
                    # np.save('/home/shanghui/dsh/storage/result/GHAP/align/{}/label_p.npy'.format(args.dataset),
                    #         psedo_labels.cpu().detach().numpy())

    logging.info(
        'Epoch: {} ACC = {} NMI = {}   ARI = {},Purity = {}, Fscore = {},Precision = {}, Recall = {}, AMI = {}'.format(
            best_epoch,
            best_scores_kmeans[0], best_scores_kmeans[1], best_scores_kmeans[2],
            best_scores_kmeans[3], best_scores_kmeans[4], best_scores_kmeans[5],
            best_scores_kmeans[6], best_scores_kmeans[7]))
    # logging.info(
    #     'Secod: ACC = {} NMI = {}   ARI = {},Purity = {}, Fscore = {},Precision = {}, Recall = {}, AMI = {}'.format(
    #
    #         second_scores[0], second_scores[1], second_scores[2],
    #         second_scores[3], second_scores[4], second_scores[5],
    #         second_scores[6], second_scores[7]))
    # logging.info(
    #     'Third: ACC = {} NMI = {}   ARI = {},Purity = {}, Fscore = {},Precision = {}, Recall = {}, AMI = {}'.format(
    #
    #         third_scores[0], third_scores[1], third_scores[2],
    #         third_scores[3], third_scores[4], third_scores[5],
    #         third_scores[6], third_scores[7]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_z', default=64, type=int, help='choose from [32, 64]')
    parser.add_argument('--lr_train', default=0.0007, type=float, help='choose from [0.0001~0.001]')
    parser.add_argument('--batch_size', default=256, type=int, help='choose from [512, 1024, 2048]')  # fix
    parser.add_argument('--n_p', default=5, type=int, help='number of positive pairs for each sample')
    # Data
    parser.add_argument('--dataset', default='ALOI-100', type=str,
                        help='choose dataset from 0-Scene15, 1-Reuters, 2-BDGP, 3-RGBD')
    parser.add_argument('--unalign_ratio', default=0.5, type=float,
                        help='originally aligned proportions in the partially view-aligned data, unalginment ratio')
    parser.add_argument('--main_view', default=1, type=int,
                        help='main view to obtain the final clustering assignments, from[0, 1]')
    # Train
    parser.add_argument('--pre_epochs', type=int, default=0)
    parser.add_argument('--con_epochs', type=int, default=200)
    parser.add_argument('--temper', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--num_cluster', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)

    args = parser.parse_args()
    star = time.time()
    main()
    print('运行时间：', time.time()-star)