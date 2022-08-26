from torch import nn


class SimpleDenseNetwork(nn.Module):
    """
    A simple linear network as the classification head of our network.
    """

    def __init__(self, n_classes, embedding_dimension):
        super(SimpleDenseNetwork, self).__init__()

        self.n_classes = n_classes
        self.embedding_dimension = embedding_dimension

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.embedding_dimension,
                out_features=512,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=3),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
