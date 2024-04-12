import os
import sys
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)
from early_stopper import *
from hin_loader import HIN
from evaluation import *
import util_funcs as uf
from config import TAHIAConfig
from TAHIA import TAHIA
import warnings
import time
import torch
import argparse

warnings.filterwarnings('ignore')
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]


def train_tahia(args, gpu_id=0, log_on=True):
    uf.seed_init(args.seed)
    uf.shell_init(gpu_id=gpu_id)
    cf = TAHIAConfig(args.dataset)

    # ! Modify config
    cf.update(args.__dict__)
    cf.dev = torch.device("cuda:0" if gpu_id >= 0 else "cpu")
    # cf.dev = torch.device("cpu")

    # ! Load Graph
    g = HIN(cf.dataset).load_mp_embedding(cf)
    #
    #g.t_info: 表示节点类型及其相对应的序号范围
    #g.r_info: 为四元组，元组前两位表示起始节点的序号范围，元组后两位表示终止节点的序号范围
    print(f'Dataset: {cf.dataset}, {g.t_info}')
    features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y = g.to_torch(cf)

    # ! Train Init
    if not log_on: uf.block_logs()
    print(f'{cf}\nStart training..')
    # cla_loss = torch.nn.NLLLoss()
    # a = len(np.where(train_y.cpu() == 1)) / len(train_y.cpu())
    # print(a)
    # credit
    cla_loss = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.06, 0.94]))).to(cf.dev)
    # other
    # cla_loss = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.3, 0.7]))).to(cf.dev)

    model = TAHIA(cf, g, args.dataset)
    model.to(cf.dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file)

    dur = []
    w_list = []
    pre_ = []
    rec_ = []
    f1_ = []
    epoch_ = []
    best_f1 = 0
    best_rec = 0
    best_pre = 0
    result = []
    for epoch in range(cf.epochs):
        # ! Train
        t0 = time.time()
        model.train()
        logits, adj_new = model(features, adj, mp_emb, epoch)
        # train_f1, train_mif1 = eval_logits(logits, train_x, train_y)
        w_list.append(uf.print_weights(model))
        # cla_loss.to('cpu')
        l_pred = cla_loss(logits[train_x].double(), train_y)
        # l_reg = cf.alpha * torch.norm(adj, 1)
        # loss = l_pred + l_reg
        loss = l_pred
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss = loss.cpu()
            loss.backward()
        optimizer.step()

        # ! Valid
        model.eval()
        with torch.no_grad():
            # logits = model.GCN(features, adj_new)
            auc, trec, tpre, acc, val_f1, val_mif1 = eval_logits(logits, val_x, val_y)
            epoch_.append('epoch' + str(epoch))
            pre_.append(round(tpre * 100, 2))
            rec_.append(round(trec * 100, 2))
            f1_.append(round(val_f1 * 100, 2))
            print('precision:{}, recall:{}, f1:{}'.format(tpre, trec, val_f1))
            result.append('Epoch' + str(epoch) + '---->' + 'precision:{}, recall:{}, f1:{}'.format(tpre, trec, val_f1))
            if best_f1 < val_f1:
                best_f1 = val_f1
                best_rec = trec
                best_pre = tpre

        dur.append(time.time() - t0)
        uf.print_train_log(epoch, dur, loss, 0, 0)

        if cf.early_stop > 0:
            if stopper.step(val_f1, model, epoch, auc, trec, tpre, acc):
                print(f'Early stopped, loading model from epoch-{stopper.best_epoch}')
                break

    if cf.early_stop > 0:
        model.load_state_dict(torch.load(cf.checkpoint_file))
    logits, _ = model(features, adj, mp_emb, 301)
    cf.update(w_list[stopper.best_epoch])
    eval_and_save(cf, logits, test_x, test_y, val_x, val_y, stopper)
    if not log_on: uf.enable_logs()
    print('####################### Epoch ##########################')
    print(epoch_)
    print('####################### Recall ##########################')
    print(rec_)
    print('####################### Precision ##########################')
    print(pre_)
    print('####################### F1 ##########################')
    print(f1_)
    print('####################### Result ##########################')
    print(result)
    print('precision:{}, recall:{}, f1:{}'.format(best_pre, best_rec, best_f1))
    
    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset = 'credit'
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train_tahia(args, gpu_id=args.gpu_id)
