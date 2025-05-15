import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def train_transform(resize,crop_size):
    transform_list = [
        transforms.Resize(size=(resize,resize)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def mask_transform(resize,crop_size):
    transform_list = [
        transforms.Resize(size=(resize,resize)),
        transforms.RandomCrop(crop_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class MaskFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(MaskFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('1')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'MaskFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()
# Basic options
#required=True(after type=)
parser.add_argument('--fore_dir', type=str, default= "/cver/yhqian/Datasets/LCG_dataset/fore",
                    help='Directory path to a batch of fore images')
parser.add_argument('--mask_dir', type=str, default= "/cver/yhqian/Datasets/LCG_dataset/mask",
                    help='Directory path to a batch of mask images')
parser.add_argument('--back_dir', type=str, default= "/cver/yhqian/Datasets/LCG_dataset/back",
                    help='Directory path to a batch of back images')
parser.add_argument('--vgg', type=str, default='/cver/yhqian/CIG/LCG-Net/models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./tf-logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=100000)          
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--im_weight', type=float, default=12000.0)
parser.add_argument('--bp_weight', type=float, default=100)        
parser.add_argument('--emb_weight', type=float, default=8000.0)
parser.add_argument('--rem_weight', type=float, default=100.0)
parser.add_argument('--tv_weight', type=float, default=0.05)
parser.add_argument('--n_threads', type=int, default=16)            #16->0 if on windows
parser.add_argument('--save_model_interval', type=int, default=10000)   #ori=10000
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)     #vgg-> Encoder, decoder-> D


network.train()
network.to(device)

re_size = 128
fore_tf = train_transform(re_size,re_size) #1th->resize, 2nd crop
back_tf = train_transform(2*re_size,re_size)    
mask_tf = mask_transform(re_size,re_size)

fore_dataset = FlatFolderDataset(args.fore_dir, fore_tf)
back_dataset = FlatFolderDataset(args.back_dir, back_tf)
mask_dataset = MaskFolderDataset(args.mask_dir, mask_tf)

fore_iter = iter(data.DataLoader(       #create iterator in dataloader
    fore_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(fore_dataset),
    num_workers= args.n_threads))            #num_workers=args.n_threads

back_iter = iter(data.DataLoader(
    back_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(back_dataset),
    num_workers= args.n_threads))

mask_iter = iter(data.DataLoader(
    mask_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(mask_dataset),
    num_workers= args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)

    fore_images = next(fore_iter).to(device)
    back_images = next(back_iter).to(device)
    mask_images = next(mask_iter).to(device)

    loss_im, loss_bp, loss_rem, loss_tv= network(fore_images, back_images, mask_images) #net return
    loss_im = args.im_weight * loss_im      #adding weights
    loss_bp = args.bp_weight * loss_bp
    loss_rem = args.rem_weight * loss_rem
    loss_tv = args.tv_weight * loss_tv
    loss = loss_im + loss_bp + loss_rem  + loss_tv

    if (i + 1) % 10 == 0:
        print('im',loss_im)
        print('bp',loss_bp)
        print('r',loss_rem)
        print('t', loss_tv)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    writer.add_scalar('loss_im', loss_im.item(), i + 1)
    writer.add_scalar('loss_bp', loss_bp.item(), i + 1)
    writer.add_scalar('loss_rem', loss_rem.item(), i + 1)
    writer.add_scalar('loss_tv', loss_tv.item(), i + 1)
    writer.add_scalar('loss_total', loss.item(), i + 1)



    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth'.format(i + 1))

        state_dict = network.FFuse.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'FFuse_iter_{:d}.pth'.format(i + 1))
writer.close()