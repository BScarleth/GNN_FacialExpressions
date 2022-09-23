import torch
from torch_geometric.loader import DataLoader
from data.coma_local_dataset import CoMA
import torch_geometric.transforms as T
from models.point_net import PointNet
from trainer.training import Trainer
from trainer.testing import testing_model
from trainer.utils import convert_dataset_sizes
import argparse


def train_predictor(args):
    train_dataset = CoMA(root=args.project_dir+"/data/", train=True, pre_transform=T.NormalizeScale(),
                         transform=T.SamplePoints(args.num_sample_points))
    test_dataset = CoMA(root=args.project_dir+"/data/", train=False, pre_transform=T.NormalizeScale(),
                        transform=T.SamplePoints(args.num_sample_points))

    print("dataset ready")
    print("train_dataset", len(train_dataset))
    print("test_dataset", len(test_dataset))

    model = PointNet(12)
    train_dataset = convert_dataset_sizes(train_dataset, 700)
    test_dataset = convert_dataset_sizes(test_dataset, 206)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=args.weight_decay)
    trainer = Trainer(args.n_iters, args.project_dir+"/trainer/trained_models/", args.print_every, args.plot_every)

    if args.training:
        trainer.train_model(train_loader, val_loader, model, optimizer, args.device)
    else:
        path = "{}/{}".format(args.project_dir+"/trainer/trained_models/", args.trained_model)
        model.test(path)

    testing_model(test_loader, model, 12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-iters',
        type=int,
        default=3,
        help='Number of training steps')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training steps')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for Adam optimizer')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for Adam optimizer')
    parser.add_argument(
        '--num-sample-points',
        type=int,
        default=3159,
        help='Number of points to sample for each face expression')
    parser.add_argument(
        '--print-every',
        type=int,
        default=1,
        help='Print frequency for loss and accuracy')
    parser.add_argument(
        '--training',
        type=bool,
        default=True,
        help='Trained the model when True, test otherwise')
    parser.add_argument(
        '--plot-every',
        type=int,
        default=1,
        help='Saving frequency for loss and accuracy to be plotted')
    parser.add_argument(
        '--project-dir',
        type=str,
        default="/home/brenda/Documents/master/thesis/Prueba_thesis/IAS_BSGT_2022-master",
        help='Path to CoMa dataset')
    parser.add_argument(
        '--trained-model',
        type=str,
        default="pointnet_model_20220619_220010",
        help='Name of the model to evaluate')

    args, _ = parser.parse_known_args()

    cuda_availability = torch.cuda.is_available()
    print("is cuda available?", cuda_availability)
    if cuda_availability:
        args.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        args.device = 'cpu'

    train_predictor(args)
