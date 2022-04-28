import sys

from torchsummary import summary

from model.resnet import resnet18

sys.path.append('./model/sourcemodel')
sys.path.append('../Cascade_Transfer_Learning/model/sourcemodel')
sys.path.append('../model/sourcemodel')
import torch
import torch.nn as nn
import os
import Build_Network

class NewModule(nn.Module):
    def __init__(self, old_cascade_net):
        super(NewModule, self).__init__()
        modules = old_cascade_net
        lis = []
        for index, i in enumerate(modules.children()):
            if isinstance(i, Build_Network.non_first_layer_cascade_Net):
                for _, j in enumerate(i.children()):
                    lis.append(j)
                    if (isinstance(j, nn.Conv2d)) | (isinstance(j, nn.Linear)):
                        lis.append(nn.ReLU(inplace=True))
                    continue
            else:
                lis.append(i)
        #             if (isinstance(i, nn.Conv2d)) | (isinstance(i, nn.Linear)):
        #                 lis.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*lis[:-7])
        self.fc = nn.Sequential(*lis[-7:-1])

    #         print('NewModule========')
    #         print(self.features)
    #         print(self.fc)
    def forward(self, x):
        out = self.features(x)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class new_module(nn.Module):
    def __init__(self, old_module, n_class, aux_in_dim, fc_in):
        super(new_module, self).__init__()
        self.old_module = old_module
        self.in_dim = aux_in_dim
        self.fc_in = fc_in
        self.n_class = n_class
        self.aux, self.fc = self.build_aux()

    def build_aux(self):
        aux = nn.Sequential(nn.Conv2d(self.in_dim, 128, kernel_size=(3, 3), padding=(3, 3), stride=1),
                            nn.MaxPool2d(2, 2, padding=0, dilation=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU())
        fc = nn.Sequential(nn.Linear(self.fc_in, 256),
                           nn.ReLU(),
                           nn.Dropout2d(0.0),
                           nn.Linear(256, 256),
                           nn.ReLU(),
                           nn.Linear(256, self.n_class))
        return aux, fc

    def forward(self, x):
        out = self.old_module(x)
        out = self.aux(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def freeze(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        for param in m.parameters():
            param.requires_grad = False


def weight_reset(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        for para in m.parameters():
            para.requires_grad = True
        m.reset_parameters()


def reset_all_parameters_dorisnet(net):
    for child in net.children():
        print('child', child)
        if isinstance(child, Build_Network.non_first_layer_cascade_Net):
            for _, c in enumerate(child.children()):
                print('c',c)
                weight_reset(c)
        else:
            weight_reset(child)


def load_dorisnet(DorisnetAddress, n_class , layer_index=11):
    model = torch.load(os.path.join(DorisnetAddress, f'layer {layer_index}/trained model'),
                       map_location='cpu').module
    if layer_index == 0:
        for id, child in enumerate(model.children()):
            if id < 3:
                freeze(child)
            else:
                weight_reset(child)

        model._modules['fc3'] = nn.Linear(256, n_class)
    else:
        target_index = layer_index * 3
        for id, child in enumerate(model._modules[str(target_index)].children()):
            if id < 3:
                freeze(child)
            else:
                weight_reset(child)
        model._modules[str(target_index)]._modules['fc3'] = nn.Linear(256, n_class)

    return model

def E2E_intermedia(E2Enet, n_class, layer_index=11):
    aux_in = [256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128]
    fc_in = {0: 445568, 1: 123008, 2: 36992, 3: 12800, 4: 6272, 5: 3200, 6: 2048, 7: 2048, 8: 2048, 9: 2048, 10: 2048}
    target = layer_index
    old_module = []
    counter = 0
    for child in E2Enet.children():
        freeze(child)
        if isinstance(child, nn.Conv2d):
            if counter == target:
                break
            else:
                counter += 1
        if not isinstance(child, Build_Network.non_first_layer_cascade_Net):
            old_module.append(child)

    return new_module(nn.Sequential(*old_module), n_class, aux_in[target - 1], fc_in[target - 1])


def load_model(**kwargs):
    layer = kwargs['layer']
    model_name = kwargs['model_name']
    DorisnetAddress = kwargs['DorisnetAddress']
    n_class = kwargs['n_class']

    if (model_name == 'CascadeNetPretrainDoris'):
        network_address = DorisnetAddress
        net = load_dorisnet(network_address, n_class, int(layer))
        net = NewModule(net)
    elif (model_name == 'E2EPretrainedDorisNet'):
        pretrained_doris_address = kwargs['pretrained_doris_address']
        network_address = DorisnetAddress

        net = load_dorisnet(network_address, n_class, 11)
        net[-1]._modules['fc3'] = nn.Linear(256, 23)
        net.load_state_dict(torch.load(os.path.join(pretrained_doris_address
                                                    , 'imagenet23_11layer_E2EDorisNet_last.pth'), map_location='cpu')[
                                'model']
                            , strict=True)
        net = E2E_intermedia(net, n_class, int(layer) + 1)
    elif (model_name == 'E2EDorisNet'):
        network_address = DorisnetAddress
        net = load_dorisnet(network_address, n_class, int(layer))
        net = NewModule(net)
    elif model_name == 'E2EResNet18':
        pretrain = kwargs['pretrain']

        net = resnet18(pretrained=pretrain)
        net.fc = nn.Linear(512, n_class)

    return net

if __name__ == '__main__':
    args = parser_options_cascade()
    args.DorisnetAddress = '/Users/juanwenwang/PycharmProjects/Cascade_Transfer_Learning/model/sourcemodel/Source Network 3'
    args.pretrained_doris_address = 'xxxxxx'
    args.pretrain = False

    # net = load_dorisnet(args, layer_index=11)
    # reset_all_parameters_dorisnet(net)
    # net.load(torch.load(os.path.join(args.pretrained_doris_address, '*.pth'))['state_dict'])
    # print(net)
    net = load_model(layer=3, model_name='CascadeNetPretrainDoris', DorisnetAddress=args.DorisnetAddress,
                     n_class=8)
    summary(net, (3, 224, 224), device='cpu')

    # from torchsummary import summary
    #
    # num_layer = 6
    # for i in range(num_layer):
    #     intermedia = E2E_intermedia(net, args.n_class, i+1)
    #     summary(intermedia, (3, 224, 224), device='cpu')
