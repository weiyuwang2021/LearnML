# PyTorch Tensors
import torch 
i = torch.LongTensor([[2, 4]])
v = torch.FloatTensor([[1, 3], [5, 7]])
w = torch.sparse.FloatTensor(i, v)  # sparse tensor

import pandas as pd

df = pd.DataFrame(
    {
        "test1": [1, 2, 3],
        "test2": [4, 5, 6],
    }
)

print(df.head())