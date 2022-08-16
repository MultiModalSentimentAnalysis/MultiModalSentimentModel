import torch
from torch import nn


class SimpleDenseNetwork(nn.Module):
    def __init__(self, n_classes, embedding_dimension):
        super(SimpleDenseNetwork, self).__init__()

        self.n_classes = n_classes
        self.embedding_dimension = embedding_dimension

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.embedding_dimension, out_features=512, ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=3),
            nn.ReLU(inplace=True),
            nn.Softmax(),
        )

    def forward(self, input_batch):
        x = input_batch
        x = self.fc(x)
        output_batch = x

        return output_batch
