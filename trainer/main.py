import torch
#from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import SamplePoints
import random
from pytorch_model_summary import summary
from data.modelnet_local_dataset import ModelNet
from data.coma_local_dataset import CoMA
import torch_geometric.transforms as T
import os.path as osp

from models.point_net import PointNet
from trainer.training import Trainer
from trainer.testing import testing_model
from trainer.utils import showPlot, plot_original_face, plot_sequence, increase_dataset, convert_dataset_sizes
import matplotlib.pyplot as plt
import os
import argparse

def train_predictor(args):

    cwd = os.getcwd()
    #dataset = ModelNet(root='{}/../dataset_pruebas/'.format(cwd),transform=SamplePoints(num=1024))
    path = "/home/brenda/Documents/master/thesis/IAS_2020_Brenda_dataset/dataset/"
    #dataset = CoMA(root='{}/../dataset/'.format(cwd), pre_transform=T.NormalizeScale())
    dataset = CoMA(root=path, pre_transform=T.NormalizeScale())

    dataset.shuffle()

    d = []
    test_data = [6, 7, 8, 9, 10, 11] # 9, 10, 11
    for element in dataset:
        if int(element.y) in test_data:
            d.append(element)
    dataset = d
    random.shuffle(dataset)
    #path = 'data/ModelNet10'
    #dataset = ModelNet(path, transform=SamplePoints(num=1024), pre_transform=T.NormalizeScale())

    print("dataset ready")
    #dataset.transform = SamplePoints(1024)
    #dataset.shuffle()
    #[15, 80]
    #dataset = increase_dataset(dataset)
    #random.shuffle(new_dataset)
    #print("No. of dataset elements:", len(new_dataset))
    #plot_original_face(dataset, 7495)

    #plot_sequence(dataset, 7445, 7485)#6215
    #dataset.transform = SamplePoints(1024)

    indx = int((len(dataset) * 80) / 100)
    train_dataset = dataset[:indx]
    test_dataset = dataset[indx:]

    train_dataset = convert_dataset_sizes(train_dataset, 480)
    test_dataset = convert_dataset_sizes(test_dataset, 30)

    print(len(train_dataset))
    print(len(test_dataset))
    #drop_last=True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) #num_workers=6
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = PointNet(6)
    #model.to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    trainer = Trainer(args.n_iters, args.print_every, args.plot_every)

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t - (r + a)  # free inside reserved

    print("total: ", t / 1e+9)
    print("reservada: ", r / 1e+9)
    print("allocada: ", a / 1e+9)
    print("free: ", f / 1e+9)
    print("---------")
    
    if args.training:
       trainer.train_model(train_loader, val_loader, model, optimizer, args.device)
       #showPlot(plot_losses)
       #showPlot(plot_acc)
    else:
       path = "/home/brenda/Documents/master/thesis/IAS_gutierrez_2022/trainer/trained_models/pointnet_model_20220528_101641"
       model.test(path, "cpu")

    testing_model(test_loader, model, 6) #test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-iters',
        type=int,
        default=6,
        help='Number of training steps')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training steps')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate for Adam optimizer')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for Adam optimizer')
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

    args, _ = parser.parse_known_args()

    cuda_availability = torch.cuda.is_available()
    print("is cuda available?", cuda_availability)
    if cuda_availability:
        args.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        args.device = 'cpu'
    args.device = 'cpu'
    train_predictor(args)



