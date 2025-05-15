#有的图片生成出来和批量的方法相比错位了，不要用
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net
from params_position import example_all
import os
#from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import random
random.seed(42)
from function import vgg_with_intermediate
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


#Camoufalge function
def camouflage(vgg, decoder, FFuse, fore, back, mask, enc_layers):
    b,c,w,h = fore.size()
    down_sam = nn.MaxPool2d((8, 8), (8, 8), (0, 0), ceil_mode=True)
    mask = down_sam(mask)
    fore_f = vgg(fore)
    back_f = vgg(back)
    fore_feats = vgg_with_intermediate(fore, enc_layers)
    back_feats = vgg_with_intermediate(back, enc_layers)

    feat = FFuse(fore_f,back_f,mask, fore_feats, back_feats)
    output = decoder(feat)
    output = output[:,:,:w,:h]
    return output

def embed(fore,mask,back,x,y):
    n_b, c_b, w_b, h_b = back.size()
    n_f, c_f, w_f, h_f = fore.size()

    mask_b = torch.zeros([n_b, 1, w_b, h_b]).to(device)
    fore_b = torch.zeros([n_b, c_b, w_b, h_b]).to(device)

    mask_b[:,:,x:w_f + x, y : h_f+y] = mask
    fore_b[:,:, x:w_f+x, y : h_f+y] = fore
    out = torch.mul(back, 1-mask_b)
    inter = torch.mul(fore_b, mask_b)
    _, c, h, w = out.shape
    # black = torch.zeros(size= (_, c, h, w), device= device)
    # black = black + inter
    # res_mask = torch.where(black != 0, 255, black)
    output = inter + out
    return output , mask_b

# Output the coordinates of the upper left corner of the camouflage region,
# the default camouflage region is in the center of the background image.
def position(fore, back):
    a_s, b_s, c_s, d_s = back.size()
    a_c, b_c, c_c, d_c = fore.size()
    x = abs((c_s - c_c) // 2)
    y = abs((d_s - d_c) // 2)
    return x,y

parser = argparse.ArgumentParser()
parser.add_argument('--use_examples', type=int, default=0, help='Use the input and positional parameters we provide. None means input by the users.')
# If input by users
parser.add_argument('--fore', type=str, default='/cver/yhqian/CIG/ACS-Net/input/final/fore', help='Foreground image.')
parser.add_argument('--mask', type=str, default='/cver/yhqian/CIG/ACS-Net/input/final/mask', help='Mask image.')
parser.add_argument('--back', type=str, default='/cver/yhqian/CIG/ACS-Net/input/final/back', help='Background image.')
parser.add_argument('--zoomSize', type=int, default=1.2, help='Zoom size.')
parser.add_argument('--Vertical', type=int, default=100, help='Move the camouflage region in the vertical direction, the larger the value, the lower the region.')
parser.add_argument('--Horizontal', type=int, default=0, help='Move the camouflage region in the horizontal direction, the larger the value, the more right the region.')
# Crop parameters
parser.add_argument('--Left', type=int, default=0)
parser.add_argument('--Right', type=int, default=-2)
parser.add_argument('--Top', type=int, default=0)
parser.add_argument('--Bottom', type=int, default=-2)

parser.add_argument('--vgg', type=str, default='/cver/yhqian/CIG/LCG-Net/models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='/cver/yhqian/CIG/ACS-Net/experiments/exp2/decoder_iter_90000.pth')
parser.add_argument('--FFuse', type=str, default='/cver/yhqian/CIG/ACS-Net/experiments/exp2/FFuse_iter_90000.pth')

# Additional options
parser.add_argument('--fore_size', type=int, default=0,
                    help='New (minimum) size for the fore image, \
                    keeping the original size if set to 0')
parser.add_argument('--back_size', type=int, default=512,
                    help='New (minimum) size for the back image, \
                    keeping the original size if set to 0')
parser.add_argument('--mask_size', type=int, default=0,
                    help='New (minimum) size for the mask image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output/final',
                    help='Directory to save the output image(s)')
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)
# Create Imgs and GT subdirectories
(output_dir / "Imgs").mkdir(exist_ok=True)
(output_dir / "GT").mkdir(exist_ok=True)

if args.use_examples:
    assert (args.use_examples>0 and args.use_examples<7)
    example = example_all[args.use_examples-1]
    fore_path = [Path(example['fore_path'])]
    mask_path = [Path(example['mask_path'])]
    back_path = [Path(example['back_path'])]

    zoomSize = example['zoomSize']
    Vertical = example['Vertical']
    Horizontal = example['Horizontal']
    # Crop
    Left = example['Left']
    Right = example['Right']
    Top = example['Top']
    Bottom = example['Bottom']
else:
    assert (args.fore)
    fore_path = Path(args.fore)
    assert (args.mask)
    mask_path = Path(args.mask)
    assert (args.back)
    back_path = Path(args.back)

    zoomSize = args.zoomSize
    Vertical = args.Vertical
    Horizontal = args.Horizontal
    Left = args.Left
    Right = args.Right + 1
    Top = args.Top
    Bottom = args.Bottom + 1




decoder = net.decoder
vgg = net.vgg
FFuse = net.FFuse(in_planes = 512)

decoder.eval()
vgg.eval()
FFuse.eval()

decoder.load_state_dict(torch.load(args.decoder))
FFuse.load_state_dict(torch.load(args.FFuse))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

enc_layers = [
    nn.Sequential(*list(vgg.children())[:4]),
    nn.Sequential(*list(vgg.children())[4: 11]),
    nn.Sequential(*list(vgg.children())[11: 18]),
    nn.Sequential(*list(vgg.children())[18: 31]),
]

vgg.to(device)
decoder.to(device)
FFuse.to(device)


fore_tf = test_transform(args.fore_size, args.crop)
back_tf = test_transform(args.back_size, args.crop)
mask_tf = test_transform(args.mask_size, args.crop)

back_list = os.listdir(back_path)

# 手动建立前景和背景的映射字典
foreground_to_background_map = {
    "ILSVRC2012_test_00000591.jpg": "00000073_(2).jpg",
    "ILSVRC2012_test_00000843.jpg": "00000323_(3).jpg",
    "ILSVRC2012_test_00001327.jpg": "00000664_(3).jpg",
    "ILSVRC2012_test_00002769.jpg": "00000489_(3).jpg",
    "ILSVRC2012_test_00005847.jpg": "00000865.jpg",
    "ILSVRC2012_test_00007113.jpg": "00000115.jpg",
    "ILSVRC2012_test_00011785.jpg": "00000515.jpg",
    "ILSVRC2012_test_00015980.jpg": "00000750.jpg",
    "ILSVRC2012_test_00030379.jpg": "00000262_(6).jpg",
    "ILSVRC2012_test_00033107.jpg": "00000053.jpg",
    "ILSVRC2012_test_00098176.jpg": "00000296_(6).jpg",
    "ILSVRC2013_test_00005144.jpg": "00000522_(3).jpg",
    "ILSVRC2013_test_00008287.jpg": "00000310_(2).jpg",
    "sun_arrqnjvtjjxexvsh.jpg": "00000218_(6).jpg",
    "sun_atgndbbztlooxmtb.jpg": "00000119.jpg",
    "sun_bdccvikqsthnkvzs.jpg": "00000329_(6).jpg",
    "sun_begbxjmgihkwgpvz.jpg": "00000538_(4).jpg",
    "sun_bssayngeipzwipnu.jpg": "00000107.jpg"
    # 添加更多的映射关系
}

not_suit = []

for img_name in os.listdir(fore_path):
    try:
        if img_name[:-4] in not_suit:
            continue
        fore_img = fore_path / Path(img_name)
        mask_img = mask_path / Path(img_name[:-4] + ".png")

        # 查找前景图像对应的背景图像
        if img_name not in foreground_to_background_map:
            print(f"没有为 {img_name} 找到对应的背景图像，跳过该图像。")
            continue
        
        back_name = foreground_to_background_map[img_name]  # 根据字典获取对应的背景图像
        back_img = back_path / Path(back_name)

        fore = Image.open(fore_img)
        back = Image.open(back_img)

        if fore.mode != "RGB":
            fore = fore.convert("RGB")      #avoid grayscale img
        if back.mode != "RGB":
            back = back.convert("RGB")
        # If the foreground is larger than the background, scale the foreground to the background size.
        tempSize = [fore.size[0] * zoomSize, fore.size[1] * zoomSize]
        if tempSize[0] > back.size[0]:
            tempSize[0] = back.size[0]
            tempSize[1] = int(tempSize[1] * back.size[0] /(fore.size[0]*zoomSize))
        if tempSize[1] > back.size[1]:
            temp = tempSize[1]
            tempSize[1] = back.size[1]
            tempSize[0] = int(tempSize[0] * back.size[1] / (temp))

        fore_tf = test_transform((int(tempSize[1]), int(tempSize[0])), args.crop)
        mask_tf = test_transform((int(tempSize[1]), int(tempSize[0])), args.crop)

        fore = fore_tf(fore)
        back = back_tf(back)
        mask = mask_tf(Image.open(mask_img))

        back = back.to(device).unsqueeze(0)
        fore = fore.to(device).unsqueeze(0)
        mask = mask.to(device).unsqueeze(0)

        mask = (mask>0).float()
        _,_,w,h =mask.shape

        x, y = position(fore, back)
        Vertical = Vertical if Vertical<=x else x
        Horizontal = Horizontal if Horizontal<=y else y
        x = x + Vertical
        y = y + Horizontal

        back_use = back[:, :, x:x + w, y:y + h]

        with torch.no_grad():
            output_pre = camouflage(vgg, decoder, FFuse, fore, back_use, mask, enc_layers)
            output_pre, black = embed(output_pre, mask, back, x, y)[0][:,:,Top:Bottom,Left:Right], embed(output_pre, mask, back, x, y)[1][:,:,Top:Bottom,Left:Right]
        output_name = output_dir.joinpath("Imgs") / Path(img_name)
        save_image(output_pre, str(output_name)[:-4] + "_back_" + back_name[:-4] + ".jpg")
        save_image(black, str(output_dir) + "/GT/" + img_name[:-4] + "_back_" + back_name[:-4] + ".png")
        print(img_name[:-4] + "_back_" + back_name[:-4] + " camouflaged.")
    except RuntimeError:
        print(img_name + "size is not suitable, please filter it.")