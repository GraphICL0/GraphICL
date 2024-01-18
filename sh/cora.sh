CUDA_VISIBLE_DEVICES_DEFAULT=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_DEFAULT python main.py --embedder graphicl --propagate True --dataset Cora --setting lt --imbalance_ratio 100
