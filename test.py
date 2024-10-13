import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel

from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--input_image', type=str, default='', help='path to the input image')  # modified
parser.add_argument('--output_dir', type=str, default='./results', help='directory for saving output image')  # modified
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

def save_image(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    # Load the model from the default path
    model = Finetunemodel(args.model)
    model = model.cuda()

    # Load the input image
    input_image_path = args.input_image
    if not os.path.exists(input_image_path):
        print(f'Error: Input image {input_image_path} does not exist.')
        sys.exit(1)
    
    # Prepare the image
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = torch.from_numpy(np.asarray(input_image).transpose(2, 0, 1)).float().unsqueeze(0).cuda()

    model.eval()
    with torch.no_grad():
        input_var = Variable(input_tensor, volatile=True).cuda()
        i, r = model(input_var)

        # Save the output image
        output_image_name = os.path.basename(input_image_path).split('.')[0] + '_output.png'
        output_image_path = os.path.join(args.output_dir, output_image_name)
        save_image(r, output_image_path)
        print(f'Processing completed. Output saved at: {output_image_path}')


if __name__ == '__main__':
    main()
