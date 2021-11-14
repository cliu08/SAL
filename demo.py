"""
Scrpt for showing the unsupervised domain adaptation using
stochastic augmented Lagrangian method.
"""


import argparse
import pathlib
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from model import config
from model.make_data import get_datasets
from model.models import Net
from model.mmd_loss import MMD_loss
from model.utils import GrayscaleToRgb, GradientReversal

from model.salm import stochatic_augmented_lagrangian


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Directory constants.
THIS_DIR = pathlib.Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
MODEL_FILE = THIS_DIR / "trained_models"

SOURCE_DATA = "mnist"

class ComparisonConfig(config.DomainAdaptConfig):
    """ Configuration for the comparison study.
    """
    def _init(self):
        super()._init()
        self.add_argument('MODEL_FILE', help='A model in trained_models')
        self.add_argument('--seed', default=1234, type=int, help="Random Seed")
        self.add_argument('--print', action='store_true')
        self.add_argument('--null', action='store_true')
        self.add_argument('--is_sal', default=True, type=bool,
            help="if the SALM is activated")



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
    # TODO: Why discriminator is here?
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(500, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    svhn, mnist = get_datasets(is_training=True)
    source_dataset = mnist if SOURCE_DATA == 'mnist' else svhn
    target_dataset = svhn if SOURCE_DATA == 'mnist' else mnist
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    # TODO: why discriminator is here?
    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)

            features_source_x = feature_extractor(source_x).view(source_x.shape[0], -1)
            features_target_x = feature_extractor(target_x).view(target_x.shape[0], -1)
            label_preds_source_x = clf(features_source_x[:source_x.shape[0]])
            label_preds_target_x = clf(features_target_x[:target_x.shape[0]])
            label_loss = F.cross_entropy(label_preds_source_x, label_y)
            # MMD loss for features.
            mmd_loss = MMD_loss(features_source_x, features_target_x)
            if args.is_sal:
                # For SALM.
                loss_dict = {'ce': label_loss, 'mmd': mmd_loss}
                stoalm_dict, loss_dict = stochatic_augmented_lagrangian(
                    stoalm_dict,
                    loss_dict,
                    epoch
                )
                loss = loss_dict['ce'] + loss_dict['mmd']
            else:
                # For fixed weight.
                loss = label_loss + mmd_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_label_accuracy += (
                label_preds_source_x.max(1)[1] == label_y
            ).float().mean().item()
        mean_loss = total_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: mean_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')

        torch.save(model.state_dict(), MODEL_FILE/'demo.pt')


if __name__ == '__main__':
    parser = ComparisonConfig('demo for salm')
    args = parser.parse_args()
    print(args)
    main(args)
