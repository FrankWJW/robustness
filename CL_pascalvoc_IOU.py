import glob
import os

import argparse
import shutil
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torchvision.datasets
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from cam.gradcam import GradCam
from ray import tune
import ray

from model.CascadeNet import build_cfg_list, cascade_net, build_e2e
from util import FileReader

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
object_categories = {name: torch.tensor(i, dtype=torch.long) for i, name in enumerate(object_categories)}


def preprocessing(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)


def ROI_2_mask(label):
    """
    converting x, y, x^ y^ to binary mask with same size from image
    return PIL image
    """

    target_transform = transforms.Compose([transforms.Resize(size=256),
                                           transforms.CenterCrop(size=224),
                                           transforms.ToTensor(), ])

    masks = []
    names = []
    height = int(label['annotation']['size']['height'])
    width = int(label['annotation']['size']['width'])

    for i in label['annotation']['object']:
        bbox = i['bndbox']
        bbox = dict(zip(bbox, map(int, bbox.values())))
        x, y, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

        l = i['name']
        mask = Image.fromarray(np.zeros((height, width)))
        draw = ImageDraw.Draw(mask)
        draw.rectangle(((x, y), (xmax, ymax)), outline="#FFFFFF", fill="#FFFFFF")

        masks.append(target_transform(mask))
        names.append(l)
    return {'masks': masks[0], 'names': names, 'labels': label}


def get_grayscale_map(grad_cam, img, label, enhancement=False):
    img_norm = preprocessing(img).unsqueeze(0)
    if torch.cuda.is_available():
        img_norm = img_norm.cuda()
    grayscale_cam = grad_cam(img_norm, target_category=label)
    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
    if enhancement:
        # taking abs(sqrt(gradcam))
        grayscale_cam = abs(np.sqrt(grayscale_cam + 1e-7))

    grayscale_cam = (grayscale_cam * 255).astype(np.uint8)
    return grayscale_cam


def get_feature_module(net, model_name):
    if (model_name == 'CL'):
        list_of_feature_module = [net.features]
        list_of_target_layer_names = [str(len(net.features)-1)]
    elif (model_name == 'E2E'):
        list_of_feature_module = [net.features]
        list_of_target_layer_names = ['15']
    else:
        raise NotImplementedError

    return list_of_feature_module, list_of_target_layer_names


def get_cam(e2e_net, cl_net, img, label):
    list_of_feature_module, list_of_target_layer_names = get_feature_module(cl_net, 'CL')
    feature_module = list_of_feature_module[-1]
    target_layer_names = list_of_target_layer_names[-1]
    grad_cam_cl = GradCam(model=cl_net, feature_module=feature_module,
                          target_layer_names=[target_layer_names],
                          use_cuda=True)  # -- > activation at last conv
    cam_cl = get_grayscale_map(grad_cam_cl, img, label, enhancement=False)

    list_of_feature_module, list_of_target_layer_names = get_feature_module(e2e_net, 'E2E')
    feature_module = list_of_feature_module[-1]
    target_layer_names = list_of_target_layer_names[-1]
    grad_cam_e2e = GradCam(model=e2e_net, feature_module=feature_module,
                           target_layer_names=[target_layer_names],
                           use_cuda=True)  # -- > activation at last conv
    cam_e2e = get_grayscale_map(grad_cam_e2e, img, label, enhancement=False)

    return cam_cl, cam_e2e


# model selection
def get_model_state_dict(df, sum_):
    model_state_dict = {}
    for i in df.trial_id.unique():
        for j in sum_.summary_path:
            if i in j:
                old = 0
                for file in glob.glob(os.path.join(j, 'checkpoint_*')):
                    new = int(file.split('checkpoint_')[1])
                    if new > old:
                        lm = df.loc[df['trial_id'] == i]['Learning Method'].iloc[0]
                        model_state_dict[f'{lm}'] = torch.load(os.path.join(file, 'checkpoint')
                                                               , map_location='cpu')['model_state_dict']
                        old = new
                    else:
                        continue
    return model_state_dict


def load_model(model_name, numconvaux):
    net = None  # init network first
    layer_cfg = build_cfg_list(capacity='small_with_dropout')
    num_layers = len(layer_cfg)
    layer_list = [str(i) for i in range(num_layers)]

    if model_name == 'CL':
        for k in layer_list:
            net = cascade_net(net, freeze=True, cur_layerid=int(k), layer_config=layer_cfg,
                              num_classes=20, dropout_p=0, numconvaux=numconvaux)
    else:
        for k in layer_list:
            net = cascade_net(net, freeze=True, cur_layerid=int(k), layer_config=layer_cfg,
                              num_classes=20, dropout_p=0, numconvaux=numconvaux)
        net = build_e2e(net)

    for param in net.parameters():
        param.requires_grad = True

    return net


def IOU(binary_cam, mask):
    """
    measuring intercetion over uniou (IOU) over a threshold.
    I1 and I2 are both binary image.
    IOU = intercept(I1, I2) / union(I1, I2).
    """
    intercept = np.sum((binary_cam & mask))
    union = np.sum((binary_cam | mask))
    return intercept / union


def get_color_cam(grayscale_cam, img):
    heatmap = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1]
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + (np.float32(img) / 255)
    cam = cam / (np.max(cam) + 1e-6)
    cam = np.uint8(255 * cam)

    return cam


def draw_bbox(img, bbox):
    draw = ImageDraw.Draw(img)
    x, y, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    draw.rectangle(((x, y), (xmax, ymax)), outline="#ff0000", width=5)
    return img


def remove_dotmodule(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    # load params
    return new_state_dict


def main(config):
    args = config['args']

    sum_ = FileReader(args.result_dir)
    df = sum_.df_all
    df['Learning Method'] = df.apply(sum_.get_lm_arch, axis=1)
    df = df.reset_index().rename(columns={'index': 'trial'})

    df = df.loc[df['trial_id'].isin(['096a0_00005', '096a0_00004'])]

    # selecting best trial
    best = []
    for i in df['Learning Method'].unique():
        for k in df['run'].unique():
            lm = df.loc[(df['Learning Method'] == i) & \
                        (df['run'] == k)]
            best.append(lm.loc[lm['AUC'] == lm.max()['AUC']])
    best = pd.concat(best)
    best = best.rename(columns={'training_size': 'Training Size'})

    # result
    result = {'img_id': [], 'obj_id': [], 'label': [], 'IOU_CL': [], 'IOU_E2E': []}

    # dataset
    target_transform = transforms.Compose([transforms.Lambda(ROI_2_mask)])
    transform = transforms.Compose([transforms.Resize(size=(224, 224))])
    test_set = torchvision.datasets.VOCDetection(root=args.root_dir, image_set='val', download=False,
                                                 transform=transform, target_transform=target_transform)
    # pbar = tqdm(total=len(test_set), ascii=True)

    model_state_dict = get_model_state_dict(best, sum_)
    # for m in model_state_dict.keys():
    #     model_state_dict[m] = remove_dotmodule(model_state_dict[m])

    cl_net = load_model('CL', best.loc[best['model_name'] == 'CL']['numconvaux'].unique().item())
    e2e_net = load_model('E2E', best.loc[best['model_name'] == 'E2E']['numconvaux'].unique().item())

    cl_net.load_state_dict(model_state_dict['CL (Conv)'], strict=True)
    e2e_net.load_state_dict(model_state_dict['E2E (Conv)'], strict=True)

    for imgindex, data in enumerate(test_set):
        img = data[0]
        num_instance = data[1]['labels']['annotation']['object'].__len__()
        for obj_index, (mask, label) in enumerate(zip(data[1]['masks'], data[1]['names'])):
            result['img_id'].append(imgindex)
            result['obj_id'].append(obj_index)
            result['label'].append(label)

            label = object_categories[label]

            # make prediction
            img_norm = preprocessing(img).unsqueeze(0)
            if torch.cuda.is_available():
                img_norm = img_norm.cuda()
                cl_net = cl_net.cuda()
                e2e_net = e2e_net.cuda()

            cl_predict = torch.sigmoid(cl_net(img_norm))[0][label].item()
            e2e_predict = torch.sigmoid(e2e_net(img_norm))[0][label].item()

            # gray_scale map
            gray_cam_cl, gray_cam_e2e = get_cam(e2e_net, cl_net, img, None) # selecting highest activation
            binary_cam_cl = gray_cam_cl >= (255 * config['threshold'])
            binary_cam_e2e = gray_cam_e2e >= (255 * config['threshold'])

            # draw box and save fig
            bbox = data[1]['labels']['annotation']['object'][obj_index]['bndbox']
            bbox = dict(zip(bbox, map(int, bbox.values())))
            if config['threshold'] == (40/255):
                color_cam_cl, color_cam_e2e = list(map(get_color_cam, [gray_cam_cl, gray_cam_e2e], [img, img]))
                size = data[1]['labels']['annotation']['size']
                size = dict(zip(size, map(int, size.values())))  # convert str to int
                color_cam_transform = transforms.Compose([transforms.ToPILImage(),
                                                          transforms.Resize(size=(size['height'], size['width']))])
                img_transform = transforms.Compose([transforms.Resize(size=(size['height'], size['width']))])
                img_and_cam_cl = color_cam_transform(color_cam_cl).copy()
                img_and_cam_e2e = color_cam_transform(color_cam_e2e).copy()
                img_orgsize = img_transform(img).copy()
                img_cam_box = list(map(draw_bbox, [img_and_cam_cl, img_and_cam_e2e, img_orgsize], [bbox, bbox, bbox]))
                for i, j in zip(img_cam_box, ['cl', 'e2e', 'img']):
                    with open(os.path.join(args.save_folder, f'{j}_{imgindex}_{label}.png'),
                              'wb') as f:
                        i.save(f)
                    f.close()

            result['IOU_CL'].append(IOU(binary_cam_cl, mask.numpy().astype(bool)))
            result['IOU_E2E'].append(IOU(binary_cam_e2e, mask.numpy().astype(bool)))

            tune.report(img_id=imgindex, obj_id=obj_index, label=label
                        , num_instance=num_instance
                        , obj_width=bbox['xmax'] - bbox['xmin'], obj_height=bbox['ymax'] - bbox['ymin']
                        , IOU_CL=IOU(binary_cam_cl, mask.numpy().astype(bool))
                        , IOU_E2E=IOU(binary_cam_e2e, mask.numpy().astype(bool))
                        , CL_predict=cl_predict
                        , E2E_predict=e2e_predict)

        # pbar.update(1)

    # pd.DataFrame(result).to_csv(args.save_csv)


if __name__ == '__main__':
    os.environ['RAY_worker_register_timeout_seconds'] = '120'
    parser = argparse.ArgumentParser()

    # ray tune
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")

    # folder
    parser.add_argument('--root_dir', type=str, default='/local/jw7u18/pascalvoc')
    parser.add_argument('--result_dir', type=str, default='/local/jw7u18/ray_results/CL_pascal')
    parser.add_argument('--save_folder', type=str, default='/local/jw7u18/ray_results/CL_pascalvoc_IOU/figs/',
                        help='folder for saving checkpoints')
    parser.add_argument('--name', type=str, default='CL_pascalvoc_IOU')

    # others
    parser.add_argument('--print_freq', type=int, default=10, help='iterations')
    parser.add_argument('--n_gpu', default=1.0, type=float)
    parser.add_argument('--num_workers', type=int, default=8)

    # ray
    parser.add_argument(
        "--ray_address",
        help="Address of Ray cluster for seamless distributed execution.")

    args, _ = parser.parse_known_args()

    if args.ray_address:
        ray.init(address=args.ray_address, _redis_password='65884528')

    try:
        shutil.rmtree(os.path.join('/local/jw7u18/ray_results', args.name))
        shutil.rmtree(args.save_folder)
    except:
        pass

    try:
        os.makedirs(args.save_folder)
    except:
        pass

    # inter_id = ['7', '12', '17', '22']
    # try:
    #     for i in inter_id:
    #         os.makedirs(os.path.join(args.save_folder, i))
    # except:
    #     pass

    # config ={
    #         "args": args,
    #         "threshold": 40 / 255
    # #        "inter_layer_id": '17
    #     }
    # main(config)

    analysis = tune.run(
        main,
        num_samples=1,
        resources_per_trial={"cpu": 4, "gpu": 1},
        mode="max",
        config={
            "args": args,
            "threshold": tune.grid_search([40 / 255, 80 / 255, 180 / 255, 200 / 255]),
            # "inter_layer_id": tune.grid_search(inter_id)
        },
        # scheduler=sche,
        name=args.name,
        resume=False,
        local_dir='/local/jw7u18/ray_results'
    )
