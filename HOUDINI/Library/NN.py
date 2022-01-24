
import json
import abc
import pathlib
import torch
import torch.nn as nn


class SaveableNNModule(nn.Module):
    def __init__(self, params_dict: dict = None):
        self.params_dict = params_dict
        super(SaveableNNModule, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    def load(self, filename):
        self.load_state_dict(torch.load(filename,
                                        map_location=torch.device('cpu')))

    def save(self, model_dir):
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        file_path = '{}/{}.pth'.format(model_dir, self.name)
        torch.save(self.state_dict(), file_path)

        out_act_key = 'output_activation'
        out_act_val = None
        if self.params_dict is not None:
            if out_act_key not in self.params_dict or \
                    self.params_dict[out_act_key] is None:
                out_act_val = 'None'
            elif self.params_dict[out_act_key] == torch.sigmoid:
                out_act_val = 'sigmoid'
            elif self.params_dict[out_act_key] == nn.Softmax:
                out_act_val = 'softmax'
            else:
                raise NotImplementedError

            self.params_dict[out_act_key] = out_act_val

            jsond = json.dumps(self.params_dict)
            f = open('{}/{}.json'.format(model_dir, self.name), 'w')
            f.write(jsond)
            f.close()


class NetDO(SaveableNNModule):
    def __init__(self,
                 name,
                 input_dim,
                 dt_name):
        super(NetDO, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.dt_name = dt_name

        self.cau_wei = nn.Parameter(data=torch.zeros(
            1, input_dim), requires_grad=True)
        self.cau_msk = nn.Parameter(data=torch.ones(
            1, input_dim), requires_grad=False)

    def forward(self, x):
        if type(x) == tuple:
            x = x[0]

        xprob = torch.sigmoid(self.cau_wei)
        out = self.cau_msk * xprob * x

        return out, xprob.detach().cpu().numpy()


class NetMLP(SaveableNNModule):
    def __init__(self,
                 name,
                 input_dim,
                 output_dim,
                 dt_name):
        super(NetMLP, self).__init__()
        self.name = name
        self.dt_name = dt_name
        self.fc0 = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        if type(x) == tuple:
            x, x_prob = x
        else:
            x_prob = None

        if self.dt_name == 'portec':
            x1 = self.fc(self.relu(self.fc0(x)))
        elif self.dt_name == 'lganm':
            x1 = self.fc(x)
        else:
            raise NotImplementedError()

        return x1, x_prob


class NetCNN(SaveableNNModule):
    def __init__(self,
                 name,
                 input_dim,
                 output_dim):
        super(NetCNN, self).__init__()
        mid_dim = (input_dim + output_dim) // 2

        kern0 = self.get_kern_size(input_dim, mid_dim)
        self.conv0 = nn.Conv1d(1, 8, kern0)

        kern1 = self.get_kern_size(mid_dim, output_dim)
        self.conv1 = nn.Conv1d(8, 1, kern1)

        self.relu = nn.ReLU()

    def get_kern_size(self,
                      input_dim,
                      output_dim,
                      stride=1,
                      padding=0):
        # easier to compute kernel size
        assert stride == 1
        kern_size = input_dim - output_dim + 2 * padding + 1
        return kern_size

    def forward(self, x):
        if type(x) == tuple:
            x, x_prob = x
        else:
            x_prob = None

        x = torch.unsqueeze(x, dim=1)
        x1 = self.conv1(self.relu(self.conv0(x)))
        x1 = torch.squeeze(x1, dim=1)
        return x1, x_prob
