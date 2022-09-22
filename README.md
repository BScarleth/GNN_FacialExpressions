# IAS_gutierrez_2022
Thesis code repository


# Dataset preparation
1- Sign up and download the`COMA_data.zip` from https://coma.is.tue.mpg.de/ (~3GB).
2- 

# Facial Expression classification

```python
# train model
python main.py --n-iters 100

# test model
python main.py --training False --trained-model "pointnet_model_{timestep}" --num-sample-points 2048

# Generated explanations with LIME 3D
python LIME_single.py --trained-model "pointnet_model_{timestep}" --sample 100 --label 8

```

## Other params
1. main
   * --batch-size
   * --learning-rate
   * --weight-decay
   * --num-sample-points: Number of points to sample from each face expression
   * --print-every: Frequency to print loss and accuracy
   * --plot-every: Frequency to plot loss and accuracy
   * --dataset-dir: Directory of the dataset. For example -> dataset/
   * --trained-models-dir: Directory to store trained models. For example -> trainer/Trained_models/
2. LIME_single
   * --dataset-dir: Directory of the dataset. For example -> dataset/
   * --basic-path: Directory where explanations are saved. For example -> explain/visu/
   * --trained-models-dir: Directory to store trained models. For example -> trainer/Trained_models/
   * --name-explanation: Name to save the explanation