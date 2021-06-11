# FRAUDRE
**Overview** 

This repository is PyTorch implementation of the method proposed in "GraphMNIST: A Graph Imbalance and Inconsistency Resistant Graph Fraud Detector".

**Requirements**  

* ```numpy``` == 1.19.5
* ```torch``` == 1.6.0
* ```scikit-learn``` == 0.23.2
* ```cuda``` == 10.1 (optional)
* ```cudnn``` == 7.6.5 (optional)

**Structure** 

* ```requirements.txt```: requirements;
* ```data/```: contains datasets ```Amazon.mat```and ```YelChi.mat``` and two demo datasets ```Amazon_demo.mat``` and ```YelpChi_demo.mat``` for your quick start;
* ```layers.py```: FRAUDRE implementations;
* ```model.py```: FRAUDRE implementaions;
* ```train.py```: training FRAUDRE and training options;  
* ```utlis.py```: input/output data, utility functions and testing GraphMNIST.

**Dataset**
* ```features.mat```: attributes of nodes;
* ```homo.mat:``` the topology formed by merging all relations (*ALL*);
* ```label.mat:``` groundtruth;
* ```net_xxx.mat:``` topology of the relation xxx.  

**Quick Start** 

To help you get started quickly, we also provide two demo datasets: ```data/Amazon_demo.mat``` and ```data/Yelp_demo.mat```;  

Run unzip ```/data/Amazon.mat.zip``` ```/data/YelpChi.mat.zip``` ```/data/Amazon_demo.mat.zip``` ```/data/YelpChi_demo.mat.zip``` to unzip the dataset; 

Set model parameters in ```train.py```;

Run python train.py to run GraphMNIST in ```CPU``` or ```GPU``` mode. 

**Baselines**  
* ```GCN```: "Semi-superbised Classification with Graph Convolutional Networks" [[Source](https://github.com/tkipf/gcn)]
* ```GAT```: "Graph Attention Networks" [[Source](https://github.com/Diego999/pyGAT)]
* ```GraphSAGE```: "Inductive Representation Learning on Large Graphs" [[Source](https://github.com/williamleif/GraphSAGE)]
* ```GEM```: "Heterogeneous Graph Neural Networks for Malicious Account Detection" [[Source](https://github.com/safe-graph/DGFraud)]
* ```FdGars```: "FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System" [[Source](https://github.com/safe-graph/DGFraud)]
* ```Player2Vec```: "Key Player Identification in Underground Forums over Attributed Heterogeneous Information Network Embedding" [[Source](https://github.com/safe-graph/DGFraud)]
* ```GraphConsis```: "Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection" [[Source](https://github.com/safe-graph/DGFraud)]
* ```CARE-GNN```: "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters" [[Source](https://github.com/YingtongDou/CARE-GNN)]
