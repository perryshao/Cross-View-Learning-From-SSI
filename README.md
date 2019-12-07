# Learning Representation from Skeletal Self-Similarities for Cross-View Action Recognition
## Introduction
The algorithm is described in the the paper: Learning Representation from Skeletal Self-Similarities for Cross-View Action Recognition. In this work, we propose a view-invariant description by formulating self-similarity images (SSIs) of human skeletons, and accordingly introduce a Multi-Stream Neural Network to learn invariant representations from SSIs of varying scales. 

## SSIs and the approach overview 
<p align="center">
  <img height="300" src="docs/teaser1.png">
</p>
<p align="center">
  <img height="300" src="docs/teaser2.png">
</p>

## Environment and installation
This repository is developed under **CUDA10.1** and **keras** in **python2.7**. The required python packages can be installed by:
```bash
pip install keras
pip install -r requirements.txt
```
## Data preprocessing

### NTU RGB+D Action Dataset
- Download the skeleton data of the [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) dataset
- Use the Matlab codes provided in `preprocessing/NTU/Matlab` to read the skeleton data and organize the training data with cross-view and cross-subject protocols. The training data are with **.mat** files. Some of the Matlab codes were made by referring to codes in the repository of the [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) dataset.
- Running Matlab codes. `Dataset_Folder` is the path of the downloaded raw skeleton data (**.skeleton** files).
 `Data_Path` is the path of the training and testing data (**.mat** files), which are generated from the raw skeleton data.
  ```bash
  read_skeletons_mat(Dataset_Folder)  
  read_skeletons_multiscale(Data_Path)
  ```
- Store the training files **.mat** in your own path.

### UESTC RGB-D Varying-view Action Dataset
To do...
### Northwestern-UCLA Multiview Action Dataset
To do...

## Models

There are a couple of models as shown in the below table.



## Running the code
Change the line `filepath = '/home/data/nturgbd_skeletons/ntu_data_mat/'` in the codes of following files for experiments to your own path of the prepared training and testing data in **Data preprocessing**. 

|              File                  |  Description                                                               |
|------------------------------------|:--------------------------------------------------------------------------:|
| `ntu-latefusion-spp-metric.py`     | MSNN <sub>late</sub> model (late fusion model) for NTU RGB+D dataset.      |
| `ntu-earlyfusion-spp-metric.py`    | MSNN <sub>early</sub> model (early fusion model) for NTU RGB+D dataset.    |
| `ntu-earlyfusion-spp-metric-c3d.py`| MSNN <sub>early</sub>-C3D model (early fusion model) for NTU RGB+D dataset.|

Notes: In MSNN <sub>early</sub>-C3D model, we use the existing pretrained C3D networks as 3D CNN branches instead of our own designed light-weight CNNs.


