# Cam-Traj-Rec
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)

This is the official implementation of the following paper: 
- Fudan Yu, Wenxuan Ao, Huan Yan, Guozhen Zhang, Wei Wu and Yong Li. [Spatio-Temporal Vehicle Trajectory Recovery on Road Network Based on Traffic Camera Video Data(in KDD 2022)](https://dl.acm.org/doi/10.1145/3534678.3539186). 

<p align="center">
<img src=".\img\framework.png" height = "" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall Framework.
</p>

## Requirements
- Python 3.7
- numpy == 1.21.3
- faiss == 1.5.3
- coloredlogs == 15.0.1
- scikit-learn == 1.0.2
- osmnx == 1.1.2
- networkx == 2.6.3
- shapely == 1.8.0
  
Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```
## File Description
### ./preprocessor/
- The directory "preprocessor" consists of road speed calculation & complementation(calculate_speed.py & train_speed.py) and road transition statistics(prior_statistics.py) based on map-matched historical trajectories.
- Due to the privacy concern, historical trajectories are not open access, thus these codes are not runnable. In substitute, the outputs of this directory which are used by directory "main" are saved as files in the directory "dataset".
### ./dataset/
- The directory "dataset" consists of the camera information, the camera records dataset(100w) which are visual embeddings of each record, and the road graph, as well as those outputs of directory "preprocessor" as mentioned above. Note that the camera records dataset is too large to be put in this repository, and you can download it at [here](https://cloud.tsinghua.edu.cn/f/d9e002b0e0ec4527b861/?dl=1).

### ./main/
- The directory "main" is the implementation of our framework, consisting of vehicle Re-ID clusering(cluster_algorithm.py) and trajectory recovery(routing.py), and the top module(run.py) that implements the spatio-temporal feedback and the iterative framework. Finally, eval.py implements the metric calculation to evaluate the clustering.
## Usage
```bash
cd ./main/
python run.py
```
## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{10.1145/3534678.3539186,
author = {Yu, Fudan and Ao, Wenxuan and Yan, Huan and Zhang, Guozhen and Wu, Wei and Li, Yong},
title = {Spatio-Temporal Vehicle Trajectory Recovery on Road Network Based on Traffic Camera Video Data},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539186},
doi = {10.1145/3534678.3539186},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {4413–4421},
numpages = {9},
keywords = {spatio-temporal modeling, vehicle trajectory recovery, urban computing},
location = {Washington DC, USA},
series = {KDD '22}
}
```