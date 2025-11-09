import torch
from mlp_przez_mymodel import MLP_Whole, TrainConfig

cfg = TrainConfig(dataset_name="mnist", epochs=1)
model = MLP_Whole(cfg)
for name, param in model.model.named_parameters():
    print(name, param.dtype)
    break
