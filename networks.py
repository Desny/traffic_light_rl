import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_output(outputs):
    with torch.no_grad():
        for m, output in enumerate(outputs):
            max_t = torch.max(torch.abs(output))
            if abs(max_t-0) < 1e-2:
                continue
            for n, t in enumerate(output):
                outputs[m][n] /= max_t
    return outputs


class DqnNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(inputs, 512)
        self.l2 = nn.Linear(512, outputs)
        c = np.sqrt(1 / inputs)
        nn.init.uniform_(self.l1.weight, -c, c)
        nn.init.uniform_(self.l1.bias, -c, c)
        nn.init.uniform_(self.l2.weight, -c, c)
        nn.init.uniform_(self.l2.bias, -c, c)
        # self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(inputs, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, outputs)
        # )

    def forward(self, x):
        # x = self.flatten(x)
        # return self.linear_relu_stack(x)
        x = x.to(device)
        # x = torch.tanh(self.l1(x))
        # x = F.softmax(self.l1(x), dim=0)
        # x = F.softmax(self.l2(x), dim=0)
        # x = F.relu(self.l2(self.l1(x)))
        x = F.leaky_relu(self.l1(x))
        x = self.l2(x)
        return x
