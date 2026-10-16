# FusedRGT: Feature-Enhanced and Implicit-Relation Graph Transformer for Social Bot Detection

FusedRGT is a graph neural network model designed for social bot detection on Twitter-like networks. 
It integrates **multimodal feature fusion** with **implicit relational structures** through a relational graph transformer. 
The model leverages **feature enhancement**, **multi-scale aggregation**, and **relation-aware edge augmentation** to improve classification performance.  

Key features:  
- Supports **node-level classification** for social bot detection  
- Incorporates **implicit edges** derived from behavioral or similarity-based heuristics  
- Multi-layer **RGT encoder** with **multi-head attention**  
- Multi-scale node feature fusion for richer representation  
- Compatible with standard Twitter datasets (e.g., TwiBot-20, TwiBot-22)  

The FusedRGT model was implemented based on the PyTorch framework, utilizing the PyTorch Geometric, scikit-learn, and Transformers libraries. 
Python version 3.12.9 and PyTorch version 2.6.0 were used. The datasets were partitioned into training, validation, and test sets with a 7:2:1 ratio. 
To prevent overfitting and enhance training efficiency, an early stopping strategy was applied during all experiments.

