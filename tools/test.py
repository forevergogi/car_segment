from models.decoder import DeepLab
import os
import torch
import logging
from PIL import Image,ImageOps
import torchvision.transforms as ttransforms
import argparse
from utils.eval import Eval
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class Tester():
    def __init__(self,args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.Eval = Eval(self.args.num_classes)
        self.model = DeepLab(output_stride=self.args.output_stride,
                             class_num=self.args.num_classes,
                             pretrained=self.args.imagenet_pretrained and self.args.pretrained_ckpt_file == None,
                             bn_momentum=self.args.bn_momentum,
                             freeze_bn=self.args.freeze_bn)
        if self.args.pretrained_ckpt_file:
            self.load_checkpoint(self.args.pretrained_ckpt_file)

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename,map_location=torch.device('cpu'))

            # self.current_epoch = checkpoint['epoch']
            # self.current_iter = checkpoint['iteration']
            state_dict = checkpoint['state_dict']
            new_dict = {}
            for key in state_dict:
                new_dict[key[7:]] = state_dict[key]
            self.model.load_state_dict(new_dict)
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.best_MIou = checkpoint['best_MIou']
            #
            # self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {},MIoU:{})\n"
            #       .format(self.args.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration'],
            #               checkpoint['best_MIou']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            self.logger.info("**First time to train**")

    def predict(self,input):
        self.logger.info('validating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            self.model.eval()
            pred = self.model(input)
        return pred

class ImageTransformer():
    def __init__(self,crop_size):
        self.crop_size = crop_size

    def _val_sync_transform(self, img):
        outsize = self.crop_size
        w, h = img.size
        if w > h:
            ow = outsize
            oh = int(1.0 * ow * h / w)
        else:
            oh = outsize
            ow = int(1.0 * oh * w / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        padh = outsize - oh if oh < outsize else 0
        padw = outsize - ow if ow < outsize else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        trans_img = self._img_transform(img)

        return img,trans_img

    def _img_transform(self, image):
        image_transforms = ttransforms.Compose([
            ttransforms.ToTensor(),
            ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        image = image_transforms(image)
        return image

if __name__ == '__main__':
    checkpoint_filepath = '../checkpoints/cityscapes/resnet101_best3.pth'
    test_image_dir = '../test_results/'
    test_image_name = '0.png'
    test_image_filepath = os.path.join(test_image_dir,test_image_name)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16],
                            help="choose from 8 or 16")
    arg_parser.add_argument('--num_classes', default=3, type=int,
                            help='num class of mask')
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply iamgenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=checkpoint_filepath,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")
    args = arg_parser.parse_args()
    tester = Tester(args)
    transformer = ImageTransformer(513)
    image = Image.open(test_image_filepath)
    img,trans_img = transformer._val_sync_transform(image)
    trans_img = torch.unsqueeze(trans_img,0)
    pred = tester.predict(trans_img)
    pred = torch.squeeze(pred,0).numpy()
    class_mat = np.argmax(pred,0) + 1
    mask_img = np.ones(shape=(3,513,513))
    for i in range(3):
        mask_img[i] = (mask_img[i] * 32 * (i+1)) * class_mat
    mask_img = mask_img.transpose(1,2,0).astype(np.uint8)
    prefix = test_image_name.replace('.png','')
    img.save(os.path.join(test_image_dir,'%s_crop.png'%prefix))
    Image.fromarray(mask_img).save(os.path.join(test_image_dir,'%s_mask.png'%prefix))
    print(np.sum(class_mat))






