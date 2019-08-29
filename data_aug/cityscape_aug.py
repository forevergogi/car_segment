from PIL import Image
import cv2
import os
import glob
from random import sample,randint
import shutil
import numpy as np

CARVANA_DIR = '../../carvana_data/'
CARVANA_IMGS = os.path.join(CARVANA_DIR,'images')
CARVANA_MASKS = os.path.join(CARVANA_DIR,'images_mask')

DATA_AUG_DIR = '../../cityscapes/'
DATA_AUG_TRAIN_IMG = os.path.join(DATA_AUG_DIR,'leftImg8bit/train/aug')
DATA_AUG_TEST_IMG = os.path.join(DATA_AUG_DIR,'leftImg8bit/test/aug')
DATA_AUG_VAL_IMG = os.path.join(DATA_AUG_DIR,'leftImg8bit/val/aug')

DATA_AUG_TRAIN_MASK = os.path.join(DATA_AUG_DIR,'3classesMask/train/aug')
DATA_AUG_TEST_MASK = os.path.join(DATA_AUG_DIR,'3classesMask/test/aug')
DATA_AUG_VAL_MASK = os.path.join(DATA_AUG_DIR,'3classesMask/val/aug')

CITY_LIST_DIR = '../datasets/city_list/'
CITY_AUG_DIR = '../datasets/city_list_aug/'

MAX_AUG_SIZE = 500

for path in [DATA_AUG_TRAIN_IMG,DATA_AUG_TEST_IMG,DATA_AUG_VAL_IMG,
             DATA_AUG_TRAIN_MASK,DATA_AUG_TEST_MASK,DATA_AUG_VAL_MASK,CITY_AUG_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


bg_rgb = (0,0,0)
car_rgb = (255,255,255) # the color value of the car class in the cityscapes dataset
colormap2label = np.zeros(256**3).astype(np.int32)
colormap2label[(car_rgb[0]*256+car_rgb[1])*256+car_rgb[2]] = 2

def label_indices(colormap, colormap2label):
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]

# extract the car name ids
car_img_names = glob.glob(os.path.join(CARVANA_IMGS,'*.jpg'))
car_ids = [name.split("/")[-1][:-7] for name in car_img_names]
car_ids = list(set(car_ids))
directions = [i for i in range(1,17)]

# Randomly pick car of a specific id and its direction
picked_names = []
total_size = 0
while total_size < MAX_AUG_SIZE and len(car_ids) > 0:
    id = car_ids.pop()
    rand = randint(4,8)
    picked = sample(directions,rand)
    picked = [id+'_%02d'%p for p in picked]
    picked_names += picked
    total_size += rand

# Copy these pictures to the specific file path
# And recolor the mask picture
total_picked = len(picked_names)
picked_indices = list(range(total_picked))
train_indices = sample(picked_indices,int(total_picked*0.7))
picked_indices = set(picked_indices)
for index in train_indices:
    picked_indices.remove(index)
picked_indices = list(picked_indices)
test_indices = sample(picked_indices,int(total_picked*0.2))
picked_indices = set(picked_indices)
for index in test_indices:
    picked_indices.remove(index)
val_indices = list(picked_indices)

train_aug_lst = []
test_aug_lst = []
val_aug_lst = []


def changeMaskColor(mask_filename,colormap2label):
    origin_img = Image.open(mask_filename).convert("RGB")
    img_arr = np.array(origin_img)
    class_img = label_indices(img_arr, colormap2label)
    image_file = Image.fromarray(class_img).convert('L')
    return image_file

for train_index in train_indices:
    train_file = picked_names[train_index]
    train_aug_lst.append(train_file)
    train_filename = train_file+'.jpg'
    train_new_filename = "aug_%s_leftImg8bit.png"%train_file
    train_maskname = train_file+'_mask.gif'
    src = os.path.join(CARVANA_IMGS,train_filename)
    dst = os.path.join(DATA_AUG_TRAIN_IMG,train_new_filename)
    if not os.path.exists(dst):
        shutil.copy(src,dst)
    mask = os.path.join(CARVANA_MASKS, train_maskname)
    new_mask = changeMaskColor(mask,colormap2label)
    train_mask_augname = "aug_%s_gtFine_labelTrainIds.png"%train_file
    # cv2.imwrite(os.path.join(DATA_AUG_TRAIN_MASK,train_mask_augname),new_mask)
    new_mask.save(os.path.join(DATA_AUG_TRAIN_MASK,train_mask_augname),)

# for test_index in test_indices:
#     test_file = picked_names[test_index]
#     test_aug_lst.append(test_file)
#     test_filename = test_file + '.jpg'
#     test_new_filename = "aug_%s_leftImg8bit.png" % test_file
#     test_maskname = test_file + '_mask.gif'
#     src = os.path.join(CARVANA_IMGS, test_filename)
#     dst = os.path.join(DATA_AUG_TEST_IMG, test_new_filename)
#     if not os.path.exists(dst):
#         shutil.copy(src, dst)
#     mask = os.path.join(CARVANA_MASKS, test_maskname)
#     new_mask = changeMaskColor(mask, colormap2label)
#     test_mask_augname = "aug_%s_gtFine_labelTrainIds.png"%test_file
#     # cv2.imwrite(os.path.join(DATA_AUG_TEST_MASK, test_mask_augname), new_mask)
#     new_mask.save(os.path.join(DATA_AUG_TEST_MASK, test_mask_augname))

for val_index in val_indices:
    val_file = picked_names[val_index]
    val_aug_lst.append(val_file)
    val_filename = val_file + '.jpg'
    val_new_filename = "aug_%s_leftImg8bit.png" % val_file
    val_maskname = val_file + '_mask.gif'
    src = os.path.join(CARVANA_IMGS, val_filename)
    dst = os.path.join(DATA_AUG_VAL_IMG, val_filename)
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    mask = os.path.join(CARVANA_MASKS, val_maskname)
    new_mask = changeMaskColor(mask, colormap2label)
    val_mask_augname = val_file + '_mask_aug.png'
    val_mask_augname = "aug_%s_gtFine_labelTrainIds.png" % val_file
    # cv2.imwrite(os.path.join(DATA_AUG_VAL_MASK, val_mask_augname), new_mask)
    new_mask.save(os.path.join(DATA_AUG_VAL_MASK, val_mask_augname),)

# Create the new city list file after augment
city_train_aug = os.path.join(CITY_AUG_DIR,'train.txt')
city_test_aug = os.path.join(CITY_AUG_DIR,'test.txt')
city_val_aug = os.path.join(CITY_AUG_DIR,'val.txt')
city_trainval_aug = os.path.join(CITY_AUG_DIR,'trainval.txt')

city_train = os.path.join(CITY_LIST_DIR,'train.txt')
city_test = os.path.join(CITY_LIST_DIR,'test.txt')
city_val = os.path.join(CITY_LIST_DIR,'val.txt')
city_trainval = os.path.join(CITY_LIST_DIR,'trainval.txt')

for (dst,src) in [(city_train_aug,city_train),
                  (city_test_aug,city_test),
                  (city_val_aug,city_val),
                  (city_trainval_aug,city_trainval)]:

    shutil.copy(src, dst)


written_mode = 'a+'
with open(city_train_aug,written_mode) as writer:
    for train in train_aug_lst:
        line = 'train_aug_%s'%train+'\n'
        writer.write(line)

with open(city_test_aug,written_mode) as writer:
    for test in test_aug_lst:
        line = 'test_aug_%s'%test+'\n'
        writer.write(line)

with open(city_val_aug,written_mode) as writer:
    for val in val_aug_lst:
        line = 'val_aug_%s'%val+'\n'
        writer.write(line)

with open(city_trainval_aug,written_mode) as writer:
    for train in train_aug_lst:
        line = 'train_aug_%s'%train+'\n'
        writer.write(line)
    for val in val_aug_lst:
        line = 'val_aug_%s'%val+'\n'
        writer.write(line)


















