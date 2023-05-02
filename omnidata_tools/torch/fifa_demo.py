import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F
import cv2
from torchvision import transforms
from external.omnidata.omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
from external.omnidata.omnidata_tools.torch.data.transforms import get_transform
import PIL

from PIL import Image


def setup_depth_model(ckpt_dir, device):
    pretrained_weights_path = os.path.join(ckpt_dir, 'omnidata_dpt_depth_v2.ckpt')  # 'omnidata_dpt_depth_v1.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def setup_normal_model(ckpt_dir, device):
    pretrained_weights_path = os.path.join(ckpt_dir, 'omnidata_dpt_normal_v2.ckpt')
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    return model

def get_omnidata_depth(model, images, device, output_dir):
    image_size = (384,384)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.ToTensor(),
                                         transforms.Normalize(mean=0.5, std=0.5)])
    depth_list = []
    for i,image in enumerate(images):
        w,h = image.size
        img_tensor = trans_totensor(image)[:3].unsqueeze(0).to(device)
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0),(h,w) , mode='bicubic').squeeze(0)
        output = output.clamp(0, 1)
        output_numpy = output.detach().cpu().numpy().squeeze()

        imgpath = os.path.join(output_dir,f"frame_{str(i).zfill(6)}.png")
        npypath = os.path.join(output_dir,f"frame_{str(i).zfill(6)}.npy")
        plt.imsave(imgpath,output_numpy,cmap='viridis')
        np.save(npypath,output_numpy)


        depth_list.append(output_numpy)
    return depth_list

def get_omnidata_normal(model,images,device,output_dir):
    image_size = (384,384)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                         get_transform('rgb', image_size=None)])

    normal_list = []
    for i,image in enumerate(images):
        w, h = image.size
        img_tensor = trans_totensor(image)[:3].unsqueeze(0).to(device)
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output, (h,w), mode='bicubic')
        output = output.clamp(0, 1)
        output_normalized = (2 * output - 1).permute(0,2,3,1)
        output_normalized = torch.nn.functional.normalize(output_normalized, p=2, dim=-1)
        output_normalized = output_normalized.detach().cpu().numpy()[0]

        output_vis = np.array(transforms.ToPILImage()(output[0]))

        imgpath = os.path.join(output_dir, f"frame_{str(i).zfill(6)}.png")
        npypath = os.path.join(output_dir, f"frame_{str(i).zfill(6)}.npy")
        plt.imsave(imgpath, output_vis)
        np.save(npypath, output_normalized)

        normal_list.append(output_normalized)
    return normal_list


if __name__ == '__main__':
    factor = 4
    imgdir_suffix = f'_{factor}'
    imgdir = os.path.join('/home/fspinola/zetong/fifa/regnerf/data/fifa_TUN-ALG', 'images' + imgdir_suffix)
    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]

    images = [Image.open(imgfile) for imgfile in imgfiles]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_dir = "pretrained_models"

    depth_model = setup_depth_model(ckpt_dir,device)
    normal_model = setup_normal_model(ckpt_dir, device)

    output_dir = '/home/fspinola/zetong/fifa/regnerf/data/fifa_TUN-ALG/'
    depth_dir = output_dir + f'omnidepths_{factor}'
    normal_dir = output_dir + f'omninormals_{factor}'
    os.makedirs(depth_dir,exist_ok=True)
    os.makedirs(normal_dir,exist_ok=True)
    get_omnidata_depth(depth_model,images,device,depth_dir)
    get_omnidata_normal(normal_model,images,device,normal_dir)