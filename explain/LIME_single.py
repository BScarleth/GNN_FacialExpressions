#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:27:59 2021

@author:
"""

import argparse
import numpy as np
import os
import torch
import sys
import open3d as o3d
import time
from data.coma_local_dataset import CoMA
import torch_geometric.transforms as T

from lime import lime_3d_remove
from torch_geometric.loader import DataLoader

from models.point_net import PointNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

categories = {
        0: 'bareteeth',  # 0
        1: 'cheeks_in',  # 1
        2: 'eyebrow',  # 2
        3: 'high_smile',  # 3
        4: 'lips_back',  # 4
        5: 'lips_up',  # 5
        6: 'mouth_down',  # 6
        7: 'mouth_extreme',  # 7
        8: 'mouth_middle',  # 8
        9: 'mouth_open',  # 9
        10: 'mouth_side',  # 10
        11: 'mouth_up',  # 11
}

def take_second(elem):
    return elem[1]

def take_first(elem):
    return elem[0]

def gen_pc_data(ori_data, segments, explain, label, args):
    color = np.zeros([ori_data.shape[0], 3])
    max_contri = 0
    min_contri = 0

    for k in explain[label]:
        if k[1] > 0 and k[1] > max_contri:
            max_contri = k[1]
        elif k[1] < 0 and k[1] < min_contri:
            min_contri = k[1]
    if max_contri > 0:
        positive_color_scale = 1 / max_contri

    else:
        positive_color_scale = 0
    if min_contri < 0:
        negative_color_scale = 1 / min_contri
    else:
        negative_color_scale = 0
    ex_sorted = sorted(explain[label], key=take_first, reverse=False)
    for i in range(segments.shape[0]):
        if ex_sorted[segments[i]][1] > 0:
            color[i][0] = ex_sorted[segments[i]][1] * positive_color_scale
        elif ex_sorted[segments[i]][1] < 0:
            color[i][2] = ex_sorted[segments[i]][1] * negative_color_scale
        else:
            color[i] = [0, 0, 0]
    pc_colored = np.concatenate((ori_data, color), axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_colored[:, 0:3])
    pc.colors = o3d.utility.Vector3dVector(pc_colored[:, 3:6])
    o3d.io.write_point_cloud(args.basic_path + args.name_explanation, pc)
    print("Generate point cloud", args.name_explanation, "successful!")


def reverse_points(points, segments, explain, start='positive', percentage=0.2):
    num_input_dims = points.shape[1]
    basic_path = "output/"
    filename = 'reversed.ply'
    if start == 'positive':
        to_rev_list = np.argsort(explain)[-int(len(explain) * percentage):]
        to_rev_list = to_rev_list[::-1]
    elif start == 'negative':
        to_rev_list = np.argsort(explain)[:int(len(explain) * percentage)]
    else:
        print('Wrong start input!')
        return points
    for i in range(len(to_rev_list)):
        segment_to_rev = to_rev_list[i]
        rev_points_index = np.argwhere(segments == segment_to_rev)
        for p in rev_points_index:
            points[p] = np.zeros([num_input_dims])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, 0:3])
    o3d.io.write_point_cloud(basic_path + filename, pc)
    print("Generate point cloud", filename, "successful!")
    return points


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--dataset_dir', type=str, default='/home/brenda/Documents/master/thesis/IAS_2020_Brenda_dataset/dataset/', help='Dataset root')
    parser.add_argument('--basic_path',
                        type=str, default="/home/brenda/Documents/master/thesis/IAS_gutierrez_2022/explain/visu",
                        help='Directory where explanations are save.')
    parser.add_argument('--trained-models-dir',
                        type=str,
                        default="/home/brenda/Documents/master/thesis/IAS_gutierrez_2022/trainer/Trained_models/",
                        help='Directory where trained models are saved.')
    parser.add_argument('--trained-model', type=str, default="pointnet_model_20220619_220010", help='Name of the model to evaluate')
    parser.add_argument('--sample-points', type=int,
                        default=3159,
                        help='Number of points to take from sample')
    parser.add_argument('--sample', type=int,
                        default=100,
                        help='Sample to explain')
    parser.add_argument('--label', type=int,
                        default=8,
                        help='Class to explain, if -1 then the predicted label will be explained')
    parser.add_argument('--name-explanation', type=str,
                        default="test_lime.ply",
                        help='Class to explain')
    return parser.parse_args()


def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index, size=sample_size)
    sampled = points[sampled_index]
    return sampled

def test(model, args):
    test_dataset = CoMA(root=args.dataset_dir, train=False, pre_transform=T.NormalizeScale(), transform=T.SamplePoints(args.sample_points))
    t = test_dataset[args.sample:(args.sample + 1)]
    train_loader = DataLoader(t, batch_size=1, shuffle=True)
    classifier = model.eval()
    classifier.to("cuda")

    current = next(iter(train_loader))
    current.to("cuda")
    print("Ground-truth Label: ", categories[int(current.y)], " : ", int(current.y))

    pred = classifier(current)
    _, pred_choice = torch.max(pred, dim=1)

    print('Predicted Label: ', categories[int(pred_choice)], " : ", int(pred_choice))
    points = torch.transpose(current.pos, 0, 1)
    return torch.unsqueeze(points, 0), pred_choice, pred, current.y


def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args = parse_args()

    '''MODEL LOADING'''
    classifier = PointNet(12)
    classifier.load_state_dict(torch.load(
        "{}/{}".format(args.trained_models_dir, args.trained_model)))
    with torch.no_grad():
        points, pred, logits, true_label = test(classifier, args)

    l = pred.detach().cpu().numpy()[0]
    points_for_exp = np.asarray(points.cpu().squeeze().transpose(1, 0))
    predict_fn = classifier.eval()
    explainer = lime_3d_remove.LimeImageExplainer(random_state=0)
    tmp = time.time()

    labels = (l,) if args.label == -1 else (args.label,)

    assert args.label == l, "Defined label {} does not match prediction. Try again, choose different label or use another sample!".format(
        args.label)
    if l != true_label:
        print("Warning: Predicted label does not match true label. Explanation will be generated for predicted class.")

    explanation = explainer.explain_instance(points_for_exp, predict_fn, labels=labels, top_labels=None, num_features=20,
                                              num_samples=1500, random_seed=0)

    print('Time consuming: ', time.time() - tmp, 's')
    gen_pc_data(points_for_exp, explanation.segments, explanation.local_exp, l, args)
    return explanation

def show_explanation(args):
    cloud = o3d.io.read_point_cloud("{}/{}".format(args.basic_path, args.name_explanation))  # Read the point cloud
    o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud


if __name__ == '__main__':
    args = parse_args()

    """GENERATE EXPLANATION"""
    exp = main(args)

    """SHOW EXPLANATION"""
    show_explanation(args)





