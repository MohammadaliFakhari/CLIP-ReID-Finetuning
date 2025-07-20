import torch
import numpy as np
import os
from utils.reranking import re_ranking
import re
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import shutil
from collections import OrderedDict
from sklearn import metrics
import pandas as pd
import cv2
import csv


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def return_distmat(distmat, q_pids, g_pids, img_paths):
    num_q, _ = distmat.shape
    q_imgs = img_paths[:num_q]
    g_imgs = img_paths[num_q:]
    unique_q_pids = sorted(list(set(q_pids)))
    unique_g_pids = sorted(list(set(g_pids)))
    q_info = defaultdict(dict)
    g_info = defaultdict(dict)
    for i in unique_q_pids:
        q_info[i] = defaultdict(list)
    for i in unique_g_pids:
        g_info[i] = defaultdict(list)
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    for i in q_imgs:
        pid, camid = map(int, pattern.search(i).groups())
        q_info[pid][camid].append(i)
    for i in g_imgs:
        pid, camid = map(int, pattern.search(i).groups())
        g_info[pid][camid].append(i)

    new_distmat = np.zeros([1428, 1428])

    new_q_pids = []
    new_q_camids = []

    row, col = 0, 0
    r = 0
    for i in range(len(unique_q_pids)):
        for k in q_info[unique_q_pids[i]]:
            c = 0
            col = 0
            new_q_pids.append(unique_q_pids[i])
            new_q_camids.append(k)
            r_step = len(q_info[unique_q_pids[i]][k])
            for j in range(len(unique_g_pids)):
                for m in g_info[unique_g_pids[j]]:
                    c_step = len(g_info[unique_g_pids[j]][m])
                    temp = distmat[row:row+r_step, col:col+c_step]
                    min_val = np.min(temp)
                    min_idx = np.unravel_index(np.argmin(temp), temp.shape)
                    x, y = min_idx  # Local coordinates in `temp`
                    if unique_q_pids[i]==1280 and unique_g_pids[j]==1280:
                        print(q_info[unique_q_pids[i]][k], g_info[unique_g_pids[j]][m])
                    new_distmat[r, c] = min_val
                    col += c_step
                    c += 1
            row += r_step
            r += 1

    return new_distmat, new_q_pids, new_q_camids



def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, img_paths, embedings, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3

    q_imgs = img_paths[:num_q]
    g_imgs = img_paths[num_q:]

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)

    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1


        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)


    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)

    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

        
class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self, img_paths, embedings):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, img_paths, embedings)
        

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
