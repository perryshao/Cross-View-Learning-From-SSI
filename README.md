# Learning Representations from Skeletal Self-Similarities for Cross-View Action Recognition
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

### Northwestern-UCLA Multiview Action Dataset
- Download the Northwestern-UCLA Multiview Action 3D dataset.
- Read the raw skeletons and build the cross-view splits. Edit `filepath` at the top of
  the script to point at your copy of the dataset; the `--view_N` arguments are resolved
  relative to it.
  ```bash
  python preprocessing/UCLA/read_skeletons_ucla_cv.py \
      --view_1 multiview_action/view_1 \
      --view_2 multiview_action/view_2 \
      --view_3 multiview_action/view_3
  ```
- Precompute the SSIs:
  ```bash
  python preprocessing/UCLA/compute_ssm_ucla_cv.py
  ```
- Matlab helpers for plotting skeletons and SSIs are in `preprocessing/UCLA/Matlab`.

### UWA3D-II Multiview Activity Dataset
- Download the UWA3D Multiview Activity II dataset.
- Read the raw skeletons and build the cross-view splits (protocol: views 1,2 vs 3,4):
  ```bash
  python preprocessing/UWA/read_skeletons_uwa_cv.py \
      --view_1 ActionsSkeleton/view_1/ --view_2 ActionsSkeleton/view_2/ \
      --view_3 ActionsSkeleton/view_3/ --view_4 ActionsSkeleton/view_4/
  ```
- Precompute the SSIs:
  ```bash
  python preprocessing/UWA/compute_ssm_uwa_cv.py
  ```

### UESTC RGB-D Varying-view Action Dataset
To do...



## Running the code
Change the `FILEPATH` constant at the top of each of the following files to your own path
of the prepared training and testing data from **Data preprocessing**.

|              File                  |  Description                                                               |
|------------------------------------|:--------------------------------------------------------------------------:|
| `ntu-latefusion-spp-metric.py`     | MSNN <sub>late</sub> model (late fusion model) for NTU RGB+D dataset.      |
| `ntu-earlyfusion-spp-metric.py`    | MSNN <sub>early</sub> model (early fusion model) for NTU RGB+D dataset.    |
| `ntu-earlyfusion-spp-metric-c3d.py`| MSNN <sub>early</sub>-C3D model (early fusion model) for NTU RGB+D dataset.|

Notes: In MSNN <sub>early</sub>-C3D model, we use the existing pretrained C3D networks as 3D CNN branches instead of our own designed light-weight CNNs.
This model additionally requires the Sports-1M pretrained weights `sports1M_weights_tf.h5`
(from [axon-research/c3d-keras](https://github.com/axon-research/c3d-keras), see its
`models/get_weights_and_mean.sh`) to be present in the working directory.

Other knobs at the top of each script: `GPUS` (passed to `keras.utils.multi_gpu_model`),
`MAX_LEN`, `CLASS_NUM`, `BATCH_SIZE` and `LAMBDA1` (the &lambda;<sub>1</sub> of Eq. (8),
set to 1e-6 as in Sec. IV-B).

### Shared modules

|              File     |  Description                                                                  |
|-----------------------|:------------------------------------------------------------------------------|
| `MetricLayer.py`      | Learnable Mahalanobis metric layer (Eq. 1-3). `MetricLayer_ForC3D` replicates the SSI into 3 channels for the pretrained C3D branch. |
| `ssi_layers.py`       | Lambda helpers that build the pairwise joint graph and reshape SSIs.          |
| `data_utils.py`       | Loading the `.mat` splits, sequence padding, and the `*_Raw_cv*.h5` cache.    |
| `viz_utils.py`        | `LrReducer` callback, confusion-matrix / training-curve / feature-map plots.  |

### Known differences between the NTU and UCLA/UWA code

The NTU scripts integrate the SSI computation into the network as a learnable
`MetricLayer`, which is the approach described in the paper. The UCLA and UWA scripts in
`preprocessing/` and `staging/ucla-uwa-training/` instead consume SSIs that are
**precomputed** by `compute_ssm_*_cv.py` and streamed through Keras generators; they
predate that integration and do not contain a `MetricLayer`. The two pipelines are not
interchangeable.

`staging/ucla-uwa-training/` holds the UCLA/UWA training scripts as they were used during
the experiments; they have not been cleaned up to the same standard as the NTU scripts.


