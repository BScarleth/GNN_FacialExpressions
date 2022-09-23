# Graph Neural Networks For Facial Expression Recognition

# Environment and dataset preparation
1- Sign up and download the`COMA_data.zip` from https://coma.is.tue.mpg.de/ (~3GB).

2- Move the zip file to the `data/raw/` directory.

3- Update the dataset_dir argument with the path where you stored the project. 
For example: `/your-path/IAS_BSGT_2022-master`.

4- Create a conda environment with python 3.8 and execute the following command:
`conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`

5- Follow the steps from https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to add the pytorch geomtric library to your environment.

We have installed pytorch-geometric using (for pytorch 1.11.0 and cuda 13):

`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html`

`pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html`

`pip install torch-geometric`

`pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html`

6- Install additional libraries manually or execute `pip install -r requirements.txt`


Considerations:

1- There might be some additional libraries or tricks you might need to applied if you work with a different environment.
Our experiments where runned using pytorch 1.11.0, cuda 13, ubuntu 20.04.0

2- The first time you run the experiments, the dataset will be processed and it might take some time to be ready.

3- The code from the LIME-3D instance level explanations was adapted from https://github.com/explain3d/lime-3d 
Tan, H., & Kotthaus, H. (2022). Surrogate Model-Based Explainability Methods for Point Cloud NNs 

# Facial Expression classification

There are three main operations you can try: trained a model from scratch, evaluated a trained model and generate instance level explanations.


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
   * --project-dir: Directory of the dataset. For example -> /your-path/IAS_BSGT_2022-master/
   * --trained-model: Name of the model to evaluate
2. LIME_single
   * --project-dir: Directory of the dataset. For example -> /your-path/IAS_BSGT_2022-master/
   * --name-explanation: Name to save the explanation
   * --sample-points: Number of points to sample from each face expression
   * --label: The index of the class to evaluate (0-11)
   * --name-explanation: Name of the explanation to be generated
   * --trained-model: Model instance to generate the explanations
   * --sample: The sample to generate the instance-level explanation

Indexes are only needed to generate level-instance explanations:
  * bareteeth: 0
  * cheeks_in: 1
  * eyebrow: 2
  * high_smile: 3
  * lips_back: 4
  * lips_up: 5
  * mouth_down: 6
  * mouth_extreme: 7
  * mouth_middle: 8
  * mouth_open: 9
  * mouth_side: 10
  * mouth_up: 11