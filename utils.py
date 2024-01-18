import numpy as np
import torch
import random
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import os.path as osp
import os
import logging
import sys
from torch_scatter import scatter_add

def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)

    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) * original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = (row_mask * col_mask).type(torch.bool)
            degree = torch.bincount(row[edge_mask]).to(row.device)
            if len(degree) < len(label):
                degree = torch.cat([degree, degree.new_zeros(len(label)-len(degree))], dim=0)
            degree = degree[cls_idx_list[i]]

            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = (row_mask * col_mask).type(torch.bool)

    train_mask = (node_mask * train_mask).type(torch.bool)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) * train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask

def split_manual(labels, c_train_num, idx_map):
    num_classes = len(set(labels.tolist()))
    c_idxs = [] 
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25 
    c_num_mat[:,2] = 55 

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]: c_train_num[i]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_train_num[i]+c_num_mat[i,1]: c_train_num[i]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def split_manual_lt(labels, idx_train, idx_val, idx_test):
    num_classes = len(set(labels.tolist()))
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels[idx_train]==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        val_lists = list(map(int,idx_val[labels[idx_val]==i]))
        test_lists = list(map(int,idx_test[labels[idx_test]==i]))
        random.shuffle(val_lists)
        random.shuffle(test_lists)

        c_num_mat[i,0] = len(c_idx)

        val_idx = val_idx + val_lists[:c_num_mat[i,1]]
        test_idx = test_idx + test_lists[:c_num_mat[i,2]]

    train_idx = torch.LongTensor(idx_train)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def split_natural(labels, idx_map):
    num_classes = len(set(labels.tolist()))
    c_idxs = [] 
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,4)).astype(int)

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        c_num = len(c_idx)

        if c_num == 3:
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
        else:
            random.shuffle(c_idx)
            c_idxs.append(c_idx)
            c_num_mat[i,0] = int(c_num*0.1) 
            c_num_mat[i,1] = int(c_num*0.1) 
            c_num_mat[i,2] = int(c_num*0.8) 
        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def accuracy(output, labels, sep_point=None, sep=None, pre=None):
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point 

    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds= output

    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def classification(output, labels, sep_point=None, sep=None):
    target_names = []
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        pred = output.max(1)[1].type_as(labels)
        for i in labels.unique():
            target_names.append(f'class_{int(i)}')

        return classification_report(labels, pred)

def confusion(output, labels, sep_point=None, sep=None):
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        
        pred = output.max(1)[1].type_as(labels)
    
        return confusion_matrix(labels, pred)

def performance_measure(output, labels, sep_point=None, sep=None, pre=None):
    acc = accuracy(output, labels, sep_point=sep_point, sep=sep, pre=pre)*100

    if len(labels) == 0:
        return np.nan
    
    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)
    
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    macro_F = f1_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    gmean = geometric_mean_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    bacc = balanced_accuracy_score(labels.cpu().detach(), output.cpu().detach())*100

    return acc, macro_F, gmean, bacc

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def refine_label_order(labels):
    print('Refine label order, Many to Few')
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    sorted_index = np.argsort(num_labels_each_class)[::-1]
    idx_map = {sorted_index[i]:i for i in range(num_labels)}
    new_labels = np.vectorize(idx_map.get)(labels.numpy())

    return labels.new(new_labels), idx_map

def normalize_adj_mx(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    import scipy.sparse as sp
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def process_adj(adj):
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj_mx(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))
    return sum_m 

def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()
    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    
    return adj

def normalize_sym(adj):
    """Symmetric-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()

    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    adj = torch.spmm(adj, deg_inv_sqrt.to_dense()).to_sparse()

    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def scheduler(epoch, curriculum_ep=500, func='convex'):
    if func == 'convex':
        return np.cos((epoch * np.pi) / (curriculum_ep * 2))
    elif func == 'concave':
        return np.power(0.99, epoch)
    elif func == 'linear':
        return 1 - (epoch / curriculum_ep)
    elif func == 'composite':
        return (1/2) * np.cos((epoch*np.pi) / curriculum_ep) + 1/2

def setupt_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")
    return logger

def set_filename(args):
    rec_with_ep_pre = 'True_ep_pre_' + str(args.ep_pre) + '_rw_' + str(args.rw) if args.rec else 'False'

    if args.im_ratio == 1: 
        results_path = f'./results/natural/{args.dataset}'
        logs_path = f'./logs/natural/{args.dataset}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/natural/{args.dataset}/({args.layer}){textname}', 'w')
        file = f'./logs/natural/{args.dataset}/({args.layer})lte4g.txt'
        
    else: 
        results_path = f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        logs_path = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer}){textname}', 'w')
        file = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer})lte4g.txt'
        
    return text, file

