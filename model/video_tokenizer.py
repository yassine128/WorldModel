import torch 

class vidTokenizer(torch.nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def train(self):
        pass