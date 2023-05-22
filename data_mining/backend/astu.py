from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import random
import os
import pdb
import json

app = Flask(__name__)
CORS(app)


class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, kernel_size):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=(1, kernel_size), stride=1,
                                           padding=(0, int((kernel_size-1)/2)), bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, kernel_size):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate, kernel_size)
            self.add_module("denselayer%d" % (i+1,), layer)


class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d((1, 2), stride=2))


class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=16, block_config=(3, 6, 12, 16, 8, 4), num_init_features=48,
                 bn_size=4, compression_rate=0.5, drop_rate=0.5, num_classes=4, linear_size=3150):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet, self).__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=(1, 7), stride=2, padding=(0, 3), bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d((1, 3), stride=2, padding=(0, 1)))
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i<=2:
                kernel_size = 5
            else:
                kernel_size = 3
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, kernel_size)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features*compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(linear_size, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, (1, 7), stride=1).view(features.size(0), -1)
#         print("features", out.shape)
        out = self.classifier(out)
        return out


def preprocess(x, y):
    num = len(x)

    xs = []
    ys = []
    key = 0.8

    for i in range(num):
        if y[i] == 0:
            xs.append(x[i])
            ys.append(0)
        elif y[i] == 1:
            xs.append(x[i])
            ys.append(1)
        elif y[i] == 2:
            xs.append(x[i])
            ys.append(2)
        elif y[i] == 3:
            r = random.randint(0, 1)
            if r > key:
                xs.append(x[i])
                ys.append(3)

    return xs, ys


# 流量归一化
def normalize(x):
    num = len(x)

    for i in range(num):
        temp = 0
        for j in range(2600):
            temp += pow(x[i][j], 2)
        temp = pow(temp, 0.5)
        for j in range(2600):
            x[i][j] /= temp

    return x


# 添加高斯噪声
def noise(x):
    num = len(x)

    for i in range(num):
        for j in range(1, 2600):
            minus = abs(x[i][j] - x[i][j - 1])
            x[i][j] += random.gauss(0, 0.1 * minus)

    return x

model = DenseNet()
checkpoint = torch.load('checkpoint.pth_den.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
pad = torch.load("./pad.pt")

@app.route('/upload', methods=['POST'])
def upload():
    print("yes")
    if 'file' in request.files:
        file = request.files['file']
        print(file.filename)
        if file.filename == '':
            result = {'success': False, 'message': 'No file selected.'}
            return jsonify(result)
        if file:
            try:
                data = []
                line = file.readline()
                while line:
                    print(line, "-----")
                    line = line.decode().split(',')
                    print(line)
                    for j in range(2600):
                        line[j] = float(line[j])
                    data.append(line)
                    line = file.readline()
                file.close()
                data = normalize(data)
                # print(data.shape)
                data = torch.tensor(data, dtype=torch.float32)
                data = data.view(1, 1, 1, -1)
                data_new = torch.cat((data, pad), 0)
                # print(data.shape)
                output = torch.softmax(model(data_new), 1)
                y_pred = output.argmax(dim=1)
                category = y_pred[0].item()
                print(output)
                name = ''
                if category == 0:
                    name = 'star'
                elif category == 1:
                    name = 'galaxy'
                elif category == 2:
                    name = 'qso'
                else:
                    name = 'unknown'
                result = {'success': True, 'message': 'success', 'name': name, 'score': '%.4f' % output[0][category].item()}
                return jsonify(result)
            except:
                print("except")
                result = {'success': False, 'message': 'Failed to parse file.'}
                return jsonify(result)
    else:
        result = {'success': False, 'message': 'Failed to parse file.'}
        return jsonify(result)

if __name__ == "__main__":
    # test
    # all = 0

    # for i in range(1000000, 1014128):
    #     f_name2 = "first_train_data/" + str(i) + ".txt"
    #     if not os.path.exists(f_name2):
    #         continue
    #     data = []
    #     f = open(f_name2, mode='r')
    #     for line in f:
    #         line = line.split(',')
    #         for j in range(2600):
    #             line[j] = float(line[j])
    #         data.append(line)
    #     f.close()
    #     data = normalize(data)
    #     data = torch.tensor(data, dtype=torch.float32)
    #     data = data.view(data.size(0), 1, 1, -1)
    #     data_new = torch.cat((data, pad), 0)
    #     # pdb.set_trace()
    #     output = model(data_new)
    #     y_pred = output.argmax(dim=1)
    #     if y_pred[0] != 0:
    #         print("ind", i, "ans", y_pred[0], "output", torch.softmax(output[0], 0))

        # if len(data) >= 25:
        #     print(i)
        #     break
        # data = noise(data)
    # print(data[-1][0][0][:])
    # print(data.shape)
    # output = model(torch.unsqueeze(data[-1], 0))
    # data1 = torch.unsqueeze(data[-1], 0)
    # data2 = torch.unsqueeze(data[-2], 0)
    # data3 = torch.unsqueeze(data[-5], 0)
    # data23 = torch.cat((data2, data3), 0)
    # torch.save(data23, "./pad.pt")
    # print("output", torch.softmax(output, 1))
    # print(output[-1])
    # print(y_pred)
    # --------------------test
    # f_name2 = "first_train_data/" + "1013233" + ".txt"
    # data = []
    # f = open(f_name2, mode='r')
    # for line in f:
    #     line = line.split(',')
    #     for j in range(2600):
    #         line[j] = float(line[j])
    #     data.append(line)
    # f.close()
    # data = normalize(data)
    # data = torch.tensor(data, dtype=torch.float32)
    # data = data.view(data.size(0), 1, 1, -1)
    # # data = torch.cat((torch.zeros((2, 1, 1, 2600)), data), 0)
    # print(data[-1][0][0][:])
    # print(data.shape)
    # output = model(data)
    # y_pred = output.argmax(dim=1)
    # print(output[-1])
    # print("output", torch.softmax(output, 1))
    # print(y_pred)
    # test
    print('run 0.0.0.0:12225')
    app.run(host='0.0.0.0', port=12225)
