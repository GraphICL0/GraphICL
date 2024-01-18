import argparse


def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help="Choose GPU number")
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Citeseer'])
    parser.add_argument('--setting', type=str, default='lt',
                        choices=['lt', 'natural', 'st'])  
    parser.add_argument('--imbalance_ratio', type=int, default=100,
                        help="Control the LT setting/ 100=0.01  50=0.02  20=0.05")
    parser.add_argument('--layer', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--cls_og', type=str, default='GNN', choices=['GNN', 'MLP'],
                        help="Wheter to user (GNN+MLP) or (MLP) as a classifier")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--ep', type=int, default=10000, help="Number of epochs to train.")
    parser.add_argument('--ep_early', type=int, default=1000, help="Early stop criterion.")
    parser.add_argument('--add_sl', type=str2bool, default=True, help="Whether to include self-loop")
    parser.add_argument('--adj_norm_1', action='store_false', default=True, help="D^(-1)A") 
    parser.add_argument('--adj_norm_2', action='store_true', default=False, help="D^(-1/2)AD^(-1/2)")
    parser.add_argument('--nhid', type=int, default=64, help="Number of hidden dimensions")
    parser.add_argument('--nhead', type=int, default=1, help="Number of multi-heads")
    parser.add_argument('--wd', type=float, default=5e-4, help="Controls weight decay")
    parser.add_argument('--num_seed', type=int, default=5, help="Number of total seeds")
    parser.add_argument('--is_normalize', action='store_true', default=False, help="Normalize features")
    parser.add_argument('--propagate', type=str2bool, default=False,
                        help="Wheter to use raw feature or propagated feature.")
    parser.add_argument('--p_k', type=int, default=10, help="Set the number of propagation to k.")
    parser.add_argument('--p_alpha', type=float, default=0.1, help="Set the ratio of residual in propagation.")

    return parser
