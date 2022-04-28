import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import transforms

from model.CascadeNet import cascade_net, build_cfg_list


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "fc" in name.lower():
                x = x.view(x.size(0), -1)
                x = module(x)
            elif "pooling" == name.lower():
                x = torch.nn.functional.adaptive_avg_pool2d(x, (28,28))
            elif "pooling_aux" == name.lower():
                x = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str,
                        default='/Users/juanwenwang/Datasets/Kvaisir/kvasir-dataset/esophagitis/0cbddd47-c7f6-474b-9e9a-b5d2429a2312.jpg',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for ResNet50 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    from model import resnet

    args = get_args()

    # net = resnet.resnet50(pretrained=True)
    # feature_module = net.layer4
    # target_layer_names = ['2']

    net = None  # init network first
    layer_cfg = build_cfg_list(capacity='small_with_dropout')
    num_layers = len(layer_cfg)
    layer_list = [str(i) for i in range(num_layers)]

    for k in layer_list:
        net = cascade_net(net, freeze=True, cur_layerid=int(k), layer_config=layer_cfg,
                          num_classes=20, dropout_p=0)

    # layer = 5
    # DorisnetAddress = './model/sourcemodel/Source Network 3'
    # model_dir = f'/Users/juanwenwang/Downloads/Kvasir_{layer}layer_CascadeNetPretrainDoris_last.pth'
    # net = load_dorisnet(DorisnetAddress, n_class=8, layer_index=layer)
    # net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
    # net = NewModule(net)
    # feature_module = net.features
    # target_layer_names = [str((int(layer)) * 4)]  # last conv index

    for param in net.parameters():
        param.requires_grad = True
    print(net)

    feature_module = net.features
    target_layer_names = ['12']  # last conv index
    grad_cam = GradCam(model=net, feature_module=feature_module, \
                       target_layer_names=target_layer_names, use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)
    print(input_img.shape)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    cv2.imwrite("cam.jpg", cam)
    cv2.imwrite("gray_cam.jpg", np.uint8(grayscale_cam * 255))

    from PIL import Image
    import cv2
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    _cutoff = 0.9  # convert to binary image
    _sizes = np.arange(10, 80, 10)

    imgdir = 'gray_cam.jpg'
    img = Image.open(imgdir).convert('L')
    img = np.array(img)

    g = Granulometry(_sizes)
    granu = g(img, cutoff=_cutoff)
    print('granulometry of {}: {}'.format(imgdir, np.mean(granu)))

    binary = g.gray2binary(img, _cutoff)
    cv2.imwrite('binarycam.jpg', binary)
