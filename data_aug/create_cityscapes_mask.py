from PIL import Image
import cv2 as cv
import os
import shutil
import glob
import numpy as np

person_rgb = np.array([11,11,11])
car_rgb = np.array([13,13,13])
bg_rgb = np.array([0,0,0])
colormap2label = np.zeros(256**3).astype(np.int32)
for i,map in enumerate([person_rgb,car_rgb]):
    colormap2label[(map[0] * 256 + map[1]) * 256 + map[2]] = i+1

def label_indices(colormap, colormap2label):
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


classes = ['person','car']
train_cities_list = ['zurich','strasbourg','weimar','aachen','tubingen','jena' \
    ,'bochum','darmstadt','dusseldorf','hamburg','cologne','monchengladbach', \
               'krefeld','ulm','hanover','stuttgart','erfurt','bremen']
test_cities_list = ['berlin','bielefeld','bonn','leverkusen','mainz','munich']
val_cities_list = ['frankfurt','lindau','munster']

CITYSCAPE_DIR = '../../cityscapes/'
ORIGIN_MASK_DIR = os.path.join(CITYSCAPE_DIR,'gtFine')
NEW_MASK_DIR = os.path.join(CITYSCAPE_DIR,'3classesMask')

# Create New Path
new_paths = []
for city in train_cities_list:
    new_paths.append(os.path.join(NEW_MASK_DIR,'train/'+city))
for city in test_cities_list:
    new_paths.append(os.path.join(NEW_MASK_DIR,'test/'+city))
for city in val_cities_list:
    new_paths.append(os.path.join(NEW_MASK_DIR,'val/'+city))
for path in new_paths:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Read the original mask images
for i,city in enumerate(train_cities_list):
    print('Processing the train data, current index is %d, the name is %s'%(i,city))
    path = 'train'
    full_path = os.path.join(ORIGIN_MASK_DIR, os.path.join(path, city))
    # img_format = '*labelTrainIds.png'
    img_format = '*labelTrainIds.png'
    label_imgs = glob.glob(full_path + '/' + img_format)
    for j,src in enumerate(label_imgs):
        print('Handling the image, its index is %d'%j)
        origin_img = Image.open(src).convert("RGB")
        img_arr = np.array(origin_img)
        class_img = label_indices(img_arr, colormap2label)
        image_file = Image.fromarray(class_img).convert('L')
        image_name = src.split('/')[-1]
        image_savepath = os.path.join(NEW_MASK_DIR, os.path.join(path, city))
        image_file.save(os.path.join(image_savepath,image_name))
        # dst = os.path.join(image_savepath, image_name)
        # shutil.copy(src, dst)

# for city in test_cities_list:
#     path = 'test'
#     full_path = os.path.join(ORIGIN_MASK_DIR, os.path.join(path, city))
#     # img_format = '*labelTrainIds.png'
#     img_format = '*labelTrainIds.png'
#     label_imgs = glob.glob(full_path + '/' + img_format)
#     for src in label_imgs:
#         origin_img = Image.open(src).convert("RGB")
#         img_arr = np.array(origin_img)
#         class_img = label_indices(img_arr, colormap2label)
#         image_file = Image.fromarray(class_img).convert('RGB')
#         image_name = src.split('/')[-1]
#         image_savepath = os.path.join(NEW_MASK_DIR, os.path.join(path, city))
#         image_file.save(os.path.join(image_savepath,image_name))
#         # dst = os.path.join(image_savepath, image_name)
#         # shutil.copy(src, dst)

for i,city in enumerate(val_cities_list):
    print('Processing the validation data, current index is %d, the name is %s' % (i, city))
    path = 'val'
    full_path = os.path.join(ORIGIN_MASK_DIR, os.path.join(path, city))
    # img_format = '*labelTrainIds.png'
    img_format = '*labelTrainIds.png'
    label_imgs = glob.glob(full_path + '/' + img_format)
    for j,src in enumerate(label_imgs):
        print('Handling the image, its index is %d' % j)
        origin_img = Image.open(src).convert("RGB")
        img_arr = np.array(origin_img)
        class_img = label_indices(img_arr, colormap2label)
        image_file = Image.fromarray(class_img).convert('L')
        image_name = src.split('/')[-1]
        image_savepath = os.path.join(NEW_MASK_DIR, os.path.join(path, city))
        image_file.save(os.path.join(image_savepath,image_name))
        # dst = os.path.join(image_savepath, image_name)
        # shutil.copy(src, dst)













