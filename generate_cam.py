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
# back_path = os.path.join("../DataSets/LandScape")
# back_imgs = os.listdir(back_path)
# print(back_imgs)

# idx = 1
# for back in back_imgs:
#     img = Image.open("../DataSets/LandScape/" + back)
#     img.save("./CODres/back/" + str(idx) + ".jpg")
#     idx += 1


# dataDir='../DataSets/coco2017'
# dataType='train2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# # 初始化标注数据的 COCO api 
# coco=COCO(annFile)
# print(coco)

# # # get all images containing given categories, select one at random
# #catIds = coco.getCatIds(catNms=[])   #catNms=['dog']
# imgIds = coco.getImgIds()
# print(len(imgIds))
# np.random.seed(1000)
# np.random.shuffle(imgIds)

# imgs = coco.loadImgs(imgIds[4300:4320])
# print(imgs)

# # 2) 使用 url, 对应关键字 "coco_url"
# idx = 4301
# for img in imgs:
#     img_name = img["coco_url"].split("/")[-1]
#     img_path = dataDir + "/" + dataType + "/" + img_name
#     I = Image.open(img_path)  
#     I.save("./CODres/fore/" + str(idx) + ".jpg")
    


#     width, height = I.size
#     dpi = 80
#     # black_img = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
#     black_img = Image.new('RGB', (width, height))
#     plt.figure(figsize=(width/dpi, height/dpi), dpi= dpi)    #create new img to avoid image overlay
#     plt.imshow(black_img, extent=[0, width, height, 0]); plt.axis('off')
#     annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     coco.showAnns(anns)
#     plt.savefig('./CODres/inter_mask/' + str(idx) + ".png", bbox_inches='tight', pad_inches=0, dpi = dpi)
#     plt.close()
#     #plt.show()
#     #print(anns)
#     idx += 1



# intermediate_mask_path = os.path.join("./CODres/inter_mask")
# mid_masks = os.listdir(intermediate_mask_path)
# print(mid_masks)
# for mid_mask in mid_masks:

#     img2 = cv2.imread("./CODres/inter_mask/" + mid_mask, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#     _, binary = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY_INV)
#     binary = cv2.bitwise_not(binary)        #convert to bitwise images(black and white)

#     cv2.imwrite("./CODres/mask_need_resize/" + mid_mask, binary)
#     # # 等待按键，然后关闭窗口
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()



# mask_need_resize_path = os.path.join("./CODres/mask_need_resize")
# To_resize_masks = os.listdir(mask_need_resize_path)
# print(To_resize_masks)

# for img_str in To_resize_masks:
#     mask_img = Image.open("./CODres/mask_need_resize/" + img_str)
#     ori_img = Image.open("./CODres/fore/" + img_str[:len(img_str) - 4] + ".jpg")
#     w1, h1 = ori_img.size
#     mask_img = mask_img.resize((w1, h1))
#     mask_img.save("./CODres/mask/" + img_str)

# # img1 = Image.open("./useless_imgs/000000000139.jpg")
# # img2 = Image.open("./useless_imgs/000000000139.png")

# # w1, h1 = img1.size
# # w2, h2 = img2.size
# # img2 = img2.resize((w1, h1))
# # img2.save("./useless_imgs/res_resize.png")
# # print(w1, h1)
# # print(img2.size)    


# mask_path = os.path.join("./CODres/mask")
# mask_imgs = os.listdir(mask_path)
# # 使用OpenCV读取图像
# for mask in mask_imgs:

#     img = cv2.imread(mask_path + "/" + mask)

#     # 转换为灰度图像
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 二值化
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     # 保存二值图像
#     cv2.imwrite("./CODres/mask2/" + mask, binary)

#     # # 使用PIL查看图像
#     # image = Image.open("./CODres/mask2/" + mask)
#     # image.show()

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
parser.add_argument('--fore', type=str, default='/cver/yhqian/Datasets/DUTS/DUTS-TE/DUTS-TE-Image', help='Foreground image.')
parser.add_argument('--mask', type=str, default='/cver/yhqian/Datasets/DUTS/DUTS-TE/DUTS-TE-Mask', help='Mask image.')
parser.add_argument('--back', type=str, default='/cver/yhqian/Datasets/LandScape', help='Background image.')
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
parser.add_argument('--output', type=str, default='multiple_output/DUTS_final',
                    help='Directory to save the output image(s)')
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

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

not_suit = []

for img_name in os.listdir(fore_path):
    try:
        if img_name[:-4] in not_suit:
            continue
        fore_img = fore_path / Path(img_name)
        mask_img = mask_path / Path(img_name[:-4] + ".png")
        back_name = random.choice(back_list)
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