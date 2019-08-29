# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 12:04 PM
# @Author  : yuhuan
# @Email   : huan.yu@versa-ai.com
# @File    : create_dataset_like_voc2012.py
# @Software: PyCharm
# @ProjectName: models
# @FileDesc:
#
# -----------------------------------------------------

import os
import glob
import PIL.Image as Image
import shutil
import numpy as np
import tqdm

batch_data_root_dir = './original'
save_dir = './human_scenery_seg_single_person'

save_JPEGImages = os.path.join(save_dir, 'JPEGImages')
save_Annotations = os.path.join(save_dir, 'Annotations')
save_SegmentationClass = os.path.join(save_dir, 'SegmentationClass')
save_SegmentationClassRaw = os.path.join(save_dir, 'SegmentationClassRaw')
save_SegmentationObject = os.path.join(save_dir, 'SegmentationObject')

save_ImageSets = os.path.join(save_dir, 'ImageSets')
save_ImageSets_train = os.path.join(save_dir, 'ImageSets', 'train.txt')
save_ImageSets_val = os.path.join(save_dir, 'ImageSets', 'val.txt')
save_ImageSets_trainval = os.path.join(save_dir, 'ImageSets', 'trainval.txt')

class_color_map_list = [
    # 第一批人像数据
    {
        "背景": [0, 0, 0],
        "人物": [192, 128, 128],
        "分割线": [255, 255, 255]
    },

    # 第二批人像数据
    {
        "背景": [0, 0, 0],
        "手机": [128, 0, 0],
        "包": [0, 128, 0],
        "帽子": [128, 128, 0],
        "杯子": [0, 0, 128],
        "太阳眼镜": [128, 0, 128],
        "书": [0, 128, 128],
        "食物": [128, 128, 128],
        "餐具": [64, 0, 0],
        "化妆品": [0, 0, 192],
        "其他": [192, 0, 0],
        "鞋子": [64, 128, 0],
        "人物": [192, 128, 128],
        "分割线": [255, 255, 255]
    },

    # 第三批人像数据，优化桌椅
    {
        "背景": [0, 0, 0],
        "人物": [128, 128, 128],
        "分割线": [255, 255, 255]
    },

    # 第四批补充数据，优化桌椅 【补充样本-datasets】
    {
        "背景": [0, 0, 0],
        "人物": [255, 255, 255],
        "分割线": [255, 255, 255]
    },

    # 第五批，卡通人像数据，二分类
    {
        "背景": [0, 0, 0],
        "人物": [128, 128, 128],
        "分割线": [255, 255, 255]
    },

    # dianwo
    {
        "背景": [0, 0, 0],
        "人物": [255, 255, 255]
    }
]


# 【类别--像素值】的映射关系
def create_color_idx():
    colormap2label_list = []
    for class_color_map in class_color_map_list:
        colormap2label = np.zeros(256 ** 3)
        for i, colormap in enumerate(class_color_map.values()):
            # # 类别【其它】归为背景
            # if colormap == [192, 0, 0]:
            #     print('类别【其它】归为背景')
            #     colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = 0
            # else:
            #     colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        colormap2label_list.append(colormap2label)
    return colormap2label_list


def create_save_dirs():
    for dir in [save_dir, save_JPEGImages, save_ImageSets, save_Annotations, save_SegmentationClass,
                save_SegmentationClassRaw, save_SegmentationObject]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

    for imgset_file_path in [save_ImageSets_train, save_ImageSets_val, save_ImageSets_trainval]:
        if os.path.exists(imgset_file_path):
            shutil.rmtree(imgset_file_path)


def resize_data(img, seg, size):
    height = img.height
    width = img.width

    max_size = max(height, width)
    resize_ratio = max_size * 1.0 / size

    img = img.resize((int(width / resize_ratio), int(height / resize_ratio)), Image.ANTIALIAS)
    seg = seg.resize((int(width / resize_ratio), int(height / resize_ratio)), Image.ANTIALIAS)

    return img, seg


def versa_label_indices(colormap, colormap2label):
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def process_src_image(new_data_name, img_data):
    if img_data.mode != 'RGB' or img_data.format != 'JPEG':
        # print('Uncorrect img color mode or format: {}'.format(img_src_path))
        pass
    if img_data.mode != 'RGB':
        img_data = img_data.convert('RGB')
    img_data.save(os.path.join(save_JPEGImages, '{}.jpg'.format(new_data_name)))


def process_seg_image(new_data_name, seg_data, colormap2label, matting_data=False):
    if seg_data.mode != 'RGB':
        seg_data = seg_data.convert('RGB')
    seg_data.save(os.path.join(save_SegmentationClass, '{}.png'.format(new_data_name)))

    seg_data_np = np.array(seg_data)
    if matting_data:
        seg_class_img = seg_data_np
    else:
        seg_class_img = versa_label_indices(seg_data_np, colormap2label)

    # # 去掉【其它】类别，【其它】类别划归到背景
    # seg_class_img[seg_class_img == 10] = 0
    seg_class_img[seg_class_img > 0] = 1

    seg_class_img = Image.fromarray(seg_class_img.astype(dtype=np.uint8))
    seg_class_img.save(os.path.join(save_SegmentationClassRaw, '{}.png'.format(new_data_name)), 'PNG')


def process_annotation(new_data_name, batch_data_dir, data_name):
    # ===============================process annotation XML ==========================
    src_xml_path = os.path.join(batch_data_dir, 'VOC', data_name + '.xml')
    new_xml_path = os.path.join(save_Annotations, '{}.xml'.format(new_data_name))
    shutil.copyfile(src_xml_path, new_xml_path)


def split_datasets(new_data_name):
    # ===============================split data, train : val == 9: 1 ==========================
    random_num = np.random.randint(0, 100)
    with open(save_ImageSets_trainval, 'a+') as trainval_f:
        trainval_f.write(str(new_data_name) + "\n")

    if random_num < 90:
        with open(save_ImageSets_train, 'a+') as train_f:
            train_f.write(str(new_data_name) + "\n")
    else:
        with open(save_ImageSets_val, 'a+') as val_f:
            val_f.write(str(new_data_name) + "\n")


def do_process(batch_data_dir_index, batch_data_dir, colormap2label=None):
    # src images
    imgs_src_dir = os.path.join(batch_data_dir, 'JPEGimages')
    print('原文件{}：{}'.format(batch_data_dir_index, imgs_src_dir))
    for img_src_path in tqdm.tqdm(glob.glob(os.path.join(imgs_src_dir, "*.*"))):
        data_name = os.path.basename(img_src_path)[:-4]
        new_data_name = 'versa-{}-{}'.format(batch_data_dir_index, data_name)

        src_img_path = glob.glob(os.path.join(imgs_src_dir, data_name + '.*'))[0]
        img_data = Image.open(src_img_path)  # 原图
        seg_data = Image.open(os.path.join(batch_data_dir, 'segmentationClass', data_name + '.png'))  # 分割图

        # size
        if img_data.width != seg_data.width or img_data.height != seg_data.height:
            print("The difference size of img and seg, name:{}, src:{}-{}, seg: {}-{}".format(
                img_src_path, img_data.width, seg_data.width, img_data.height, seg_data.height
            ))
            continue

        # img_data, seg_data = resize_data(img_data, seg_data, 513)
        process_src_image(new_data_name, img_data)
        process_seg_image(new_data_name, seg_data, colormap2label, matting_data=True)
        # process_annotation(new_data_name, batch_data_dir, data_name)
        split_datasets(new_data_name)

        # print("完成图片{}".format(new_data_name))


def main():
    create_save_dirs()  # 创建保存结果的目录
    colormap2label_0, colormap2label_1, colormap2label_2, colormap2label_3, colormap2label_4, colormap2label_5 = create_color_idx()

    batch_data_list = glob.glob(os.path.join(batch_data_root_dir, "*dataset*"))
    for i in range(len(batch_data_list)):
        process_dir_path = batch_data_list[i]
        if os.path.isdir(process_dir_path):
            # if process_dir_path.__contains__('201904'):
            #     do_process(i, process_dir_path, colormap2label_0)
            # elif process_dir_path.__contains__('201905'):
            #     do_process(i, process_dir_path, colormap2label_1)
            # elif process_dir_path.__contains__('494张人像分割数据'):
            #     do_process(i, process_dir_path, colormap2label_2)
            # elif process_dir_path.__contains__('【不需要了】补充样本sits'):
            #     do_process(i, process_dir_path, colormap2label_3)
            # elif process_dir_path.__contains__('卡通seg二分类'):
            #     do_process(i, process_dir_path, colormap2label_4)
            # elif process_dir_path.__contains__('DianWo'):
            #     do_process(i, process_dir_path)
            if process_dir_path.__contains__('DianWo'):
                do_process(i, process_dir_path)
            elif process_dir_path.__contains__('single-person'):
                do_process(i, process_dir_path)
        else:
            print("Not a dir:" + batch_data_list[i])


if __name__ == '__main__':
    main()
