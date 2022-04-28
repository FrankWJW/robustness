import torch
import torch.nn as nn



class CascadeNet(nn.Module):
    def __init__(self, features, in_channel, num_classes=1000, init_weights=False, add_aux=True
                 , batch_norm=True, kernel_size=2, stride=2, aux_capacity=128, numconvaux=1, **kwarg):
        super(CascadeNet, self).__init__()
        self.cur_layer = kwarg['cur_layerid']
        self.features = features
        # self.pooling = nn.AdaptiveMaxPool2d((20, 20))
        self.n_class = num_classes
        self.in_channels = in_channel
        self.add_aux = add_aux
        self.batch_norm = batch_norm
        self.kernel_size = kernel_size
        self.stride = stride

        if numconvaux:
            n = aux_capacity
        else:
            n = self.in_channels
        self.aux = aux_classifier(n, in_channel=self.in_channels, add_aux=add_aux
                                  , numconvaux=numconvaux, batch_norm=batch_norm, dropout_p=kwarg['dropout_p'])
        self.pooling_aux = nn.AdaptiveAvgPool2d((5, 5))
        self.fc = nn.Sequential(
            nn.Linear(n * 5 * 5, self.n_class),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        # out = self.pooling(out)
        out = self.aux(out)
        out = self.pooling_aux(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def aux_classifier(out_channel=256, **kwargs):
    layer = []
    k = kwargs['numconvaux']
    in_channel = kwargs['in_channel']
    if kwargs['add_aux']:
        for i in range(k):
            conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=3)
            if kwargs['batch_norm']:
                layer += [conv2d, nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True), nn.Dropout(kwargs['dropout_p'])]
            else:
                layer += [conv2d, nn.ReLU(inplace=True), nn.Dropout(kwargs['dropout_p'])]
            in_channel = out_channel

    return nn.Sequential(*layer)


def get_in_channel(_cfg):
    return [_cfg[k][1] for k in _cfg.keys()]


def make_layers(cfg, batch_norm=True, kerner_size=2, stride=2, dropout_p=0.5, **kwargs):
    layers = []
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=kerner_size, stride=stride)]
        elif v == 'D':
            layers += [nn.Dropout(dropout_p)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=3)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


def build_cfg(**kwargs):
    r"""
    :param layer_config: list of output channels e.g.
                    [64, (128,D), 64] -> [1st layer output, (2nd layer out, dropout layer), 3rd layer out]
    :return: dict
    """
    in_channel = 3  # for RGB images
    layer_config = kwargs['layer_config']

    cfg = dict()
    for l, i in enumerate(layer_config):
        if type(i) is tuple:
            in_out = [in_channel]
            for x in i:
                in_out.append(x)
            in_channel = i[0]
        else:
            in_out = [in_channel, i]
            in_channel = i
        cfg[l] = in_out

    return cfg


def concact_sequential(s1, s2, freeze):
    if freeze:
        for param in s1.parameters():
            param.requires_grad = False
    s_cat = torch.nn.Sequential(*(list(s1) + list(s2)))
    return s_cat


def step(net, new_feature, inchannel, freeze=False, **kwargs):
    """
    net adding new layer towards a new network
    :param net: nn.Module
    :return: new net, nn.Module
    """
    old_features = list(net.children())[0]
    new_features = concact_sequential(old_features, new_feature, freeze=freeze)
    return CascadeNet(new_features, inchannel, **kwargs)


def cascade_net(model=None, freeze=True, **kwargs):
    cfg = build_cfg(**kwargs)
    k = kwargs['cur_layerid']
    feat = make_layers(cfg[k], **kwargs)
    inchannel = get_in_channel(cfg)[int(k)]

    if k == 0:
        model = CascadeNet(feat, inchannel, **kwargs)
    else:
        model = step(model, feat, inchannel, freeze=freeze, **kwargs)
    return model


def build_cfg_list(capacity='vgg16'):
    if capacity == 'vgg16':
        return [64, (64, 'M'), 128, (128, 'M'), 256, 256, (256, 'M'), 512, 512, (512, 'M'), 512, 512, (512, 'M')]
    elif capacity == 'small':
        return [(256, 'M'), (256, 'M'), (256, 'M'), (256, 'M'), (256, 'M'), (256, 'M')]
    elif capacity == 'small_with_dropout':
        return [(256, 'M', 'D'), (256, 'M', 'D'), (256, 'M', 'D'), (256, 'M', 'D'), (256, 'M', 'D'), (256, 'M', 'D')]
    elif capacity == 'greedylayerwise':
        return [256, (256, 'M'), (256, 'M'), 512, 512]
    elif capacity == 'greedylayerwiseimagenet':
        return [256, (256, 'M'), (256, 'M'), (512, 'M'), 512, (512, 'M'), 512, 512]
    elif capacity == 'greedylayerwiseimagenetdropout':
        return [(256, 'D'), (256, 'D', 'M'), (256, 'M'), (512, 'M'), 512, (512, 'M'), 512, 512]
    elif capacity == 'chexpert':
        return [64, (64, 'M'), 128, (128, 'M'), 256, (256, 'M'), 512, (512, 'M'), 512, (512, 'M')]


def build_e2e(net):
    """
    Build Cascade network into E2E.
    all layer would be allow to update weights.
    :param net:
    :return: net_e2e
    """
    for param in net.parameters():
        param.requires_grad = True
    return net


if __name__ == '__main__':
    from torchsummary import summary
    # from temperature_scaling import ModelWithTemperature

    net = None

    layer_cfg = build_cfg_list(capacity='small_with_dropout')
    num_layers = len(layer_cfg)
    layer_list = [str(i) for i in range(num_layers)]

    for k in layer_list:
        net = cascade_net(net, freeze=True, cur_layerid=int(k), layer_config=layer_cfg,
                          num_classes=10, dropout_p=0.5, numconvaux=1)
        # net = ModelWithTemperature(net)

        # net = net.model
        print(net.cur_layer)
        summary(net, (3, 32, 32))

    print(net)