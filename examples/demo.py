"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""

'''
https://raw.githubusercontent.com/jvanvugt/pytorch-domain-adaptation/master/revgrad.py
'''


import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from data import MNISTM
from models import Net
from utils import GrayscaleToRgb, GradientReversal

from ..SAL import sal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from mmd_loss import MMD_loss  # todos !!!!
'''
def MMD_loss(features_source_x, features_target_x):

    # compute the mmd loss
    mmd_loss = F.mmd(features_source_x, features_target_x)


    return mmd_loss
'''


import config

class ComparisonConfig(config.DomainAdaptConfig):
    """ Configuration for the comparison study
    """

    _algorithms = ['adv', 'vada', 'dann', 'assoc', 'coral', 'teach','mmd','stoalm']
    _algo_names = [
        "Adversarial Domain Regularization",
        "Virtual Adversarial Domain Adaptation",
        "Domain Adversarial Training",
        "Associative Domain Adaptation",
        "Deep Correlation Alignment",
        "Self-Ensembling",
        "MMD",
        "Sto_ALM"
    ]

    def _init(self):
        super()._init()
        self.add_argument('--seed', default=1234, type=int, help="Random Seed")
        self.add_argument('--print', action='store_true')
        self.add_argument('--null', action='store_true')

        for arg, name in zip(self._algorithms, self._algo_names):
            self.add_argument('--{}'.format(arg), action='store_true', help="Train a model with {}".format(name))
#            print(name)


def main(args):

    stoalm_dict = {'lamb': [args.lamb], 'rho':[args.rho], 'bv':[], 'add_loss_func':[],\
                    'epsilon': args.epsilon, 'sigma':args.sigma, 'sigma_ls':[], \
                    'lamb_max':args.lamb_max, 'tau':args.tau, 'gamma':args.gamma,\
                    'fix_weight':args.fix_weight, 'loss_func_weight':args.loss_func_weight,\
                    'stoalm_optim':args.stoalm_optim.lower()}


    model = Net().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))
    feature_extractor = model.feature_extractor
    clf = model.classifier
    '''    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)
    '''


    half_batch = args.batch_size // 2
    source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    
    target_dataset = MNISTM(train=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                features_source_x = feature_extractor(source_x).view(source_x.shape[0], -1)
                features_target_x = feature_extractor(target_x).view(target_x.shape[0], -1)
                #domain_preds = discriminator(features).squeeze()
                label_preds_source_x = clf(features_source_x[:source_x.shape[0]])
                label_preds_target_x = clf(features_target_x[:target_x.shape[0]])
                
                #domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = F.cross_entropy(label_preds_source_x, label_y)
                # MMD loss for features
                features_source_x = feature_extractor(source_x).view(source_x.shape[0], -1)
                mmd_loss = MMD_loss(features_source_x, features_target_x)
                # for fixed weight
                loss = label_loss + mmd_loss
                # for SALM
                #  stoalm_dict, parameters for salm  !!!!! --- todos
                loss_dict = {'ce':label_loss, 'mmd':mmd_loss}
                stoalm_dict, loss_dict = sal(stoalm_dict, loss_dict)
                loss = loss_dict['ce'] + loss_dict['mmd']

                optim.zero_grad()
                loss.backward()
                optim.step()
        #  !!!! todos !!!
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')

        torch.save(model.state_dict(), 'demo.pt')


if __name__ == '__main__':
    parser = ComparisonConfig('demo fos salm')
    args = parser.parse_args()
    print(args)


    main(args)