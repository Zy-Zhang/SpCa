import argparse
from torch import cuda


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', metavar='EXPORT_DIR', help='destination where trained network should be saved')
    parser.add_argument('--training-dataset', default='GLDv2', help='training dataset: (default: GLDv2)')
    parser.add_argument('--imsize', default=1024, type=int, metavar='N', help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num-epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
    parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N', help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
    parser.add_argument('--update-every', '-u', default=1, type=int, metavar='N', help='update model weights every N batches, used to handle really large batches, ' + 'batch_size effectively becomes update_every x batch_size (default: 1)')
    parser.add_argument('--resume', default=None, type=str, metavar='FILENAME', help='name of the latest checkpoint (default: None)')

    parser.add_argument('--warmup-epochs', type=int, default=0, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--val-epoch', type=int, default=1)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--warmup-lr', type=float, default=0, help='Initial warmup learning rate')
    parser.add_argument('--base-lr', type=float, default=1e-6)
    parser.add_argument('--final-lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29324')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clip_max_norm', type=float, default=0)

    parser.add_argument('--model', type=str, default='solar')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--tau', type=int, default=32)
    parser.add_argument('--local-dim', type=int, default=128)
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--split', type=float, default=0.8)
    
    # parser.add_argument('--dist-measure', action='store_true')
    parser.add_argument('--combine', type=str, default='cro') # 'cro' / 'cat' / 'orth' / 'had' / 'fur'
    # parser.add_argument('--mode', type=str, default='qxx')
    parser.add_argument('--codebook-size', type=int, default=16)
    # parser.add_argument('--T', type=int, default=6)
    # parser.add_argument('--scale', type=float, default=1.)
    # parser.add_argument('--loss-attn', action='store_true')
    # parser.add_argument('--self-out', action='store_true')
    # parser.add_argument('--stage', type=str, default='s4') # 's3' / 's4' / 's3+s4'
    # parser.add_argument('--stage', action='store_true')
    # parser.add_argument('--test', action='store_true')
    # parser.add_argument('--pool', type=str, default='gem')
    # parser.add_argument('--self-attn-b', action='store_true')
    # parser.add_argument('--self-attn-a', action='store_true')
    # parser.add_argument('--drop', action='store_true')
    parser.add_argument('--multi', type=int, default=3)
    # parser.add_argument('--dilation', type=int, default=2)
    # parser.add_argument('--layer-norm', action='store_true')
    # parser.add_argument('--batch-norm', action='store_true')
    # parser.add_argument('--self-local', action='store_true')
    # parser.add_argument('--stop-gradient', action='store_true')
    # parser.add_argument('--slattn', action='store_true')
    parser.add_argument('--outputdim', type=int, default=2048)
    # parser.add_argument('--fire', action='store_true')
    parser.add_argument('--pretrained', type=str, default='v1') # 'filip', 'v1', or 'v2'
    # parser.add_argument('--softmax-revise', action='store_true')
    # parser.add_argument('--dolg-weight', action='store_true')
    # parser.add_argument('--dist-encode', action='store_true')
    # parser.add_argument('--revise', action='store_true') # if it is True, then we apply L1 normalize for \alpha and give 1/WH weight for previous \mu
    # parser.add_argument('--addnorm', action='store_true') # add layernorm after each iteration to update the templates
    # parser.add_argument('--gmm', action='store_true') # utilize GMM: \alpha = 0.9, \pi = 1/K, \cov = diagonal \sigma, \mu = randn(K,D)
    # parser.add_argument('--gamma-r', action='store_true') # if revise the gamma to correct version
    # parser.add_argument('--dolg-aspp', action='store_true') # if revise the fuser to correct version
    # parser.add_argument('--deconv', action='store_true') # if revise the fuser to correct version
    # parser.add_argument('--kernel-size', type=int, default=5)
    # parser.add_argument('--aspp', action='store_true')
    # parser.add_argument('--non-ln', action='store_true')
    # parser.add_argument('--de-dilation', action='store_true')
    # parser.add_argument('--non-gamma-decay', action='store_true')
    # parser.add_argument('--V', type=float, default=0.)
    # parser.add_argument('--tensorboard', type=str, default='run')
    # parser.add_argument('--kmeans-plus-plus', action='store_true')
    # parser.add_argument('--gamma', type=float, default=0.5)
    # parser.add_argument('--sci', type=str, default='gmm')
    parser.add_argument('--num-prompt', type=int, default=3)
    args = parser.parse_args()
    return args
