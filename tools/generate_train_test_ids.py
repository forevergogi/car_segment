import glob
import os
from random import choice

if __name__ == '__main__':
    '''
    We split the dataset into 2 parts called train,test. And the ratio is 7:1.
    In the Carvana dataset, each car has 16 images which are taken from 16 directions.
    For each car, randomly choose 2 different directions for the eval and 2 for the test,
    the rest are assigned to the train data.
    Finally, write the car and direction ids to the text files.
    '''
    DATA_PATH = '../../carvana_data/images/'
    TEXT_PATH = '../../carvana_data/'
    images = glob.glob(os.path.join(DATA_PATH,'*.jpg'))
    car_ids = [img[len(DATA_PATH):-7]for img in images]
    unique_ids = []
    for id in car_ids:
        if id not in unique_ids:
            unique_ids.append(id)
    train_imgs = []
    test_imgs = []
    for u_id in unique_ids:
        total_dirs = set(range(1,17))
        test_dirs = []
        for _ in range(2):
            rand_dir = choice(list(total_dirs))
            test_dirs.append(rand_dir)
            total_dirs.remove(rand_dir)
        train_dirs = list(total_dirs)
        train_dirs.sort()
        test_dirs.sort()
        for dir in train_dirs:
            train_imgs.append(('%s_%02d')%(u_id,dir))
        for dir in test_dirs:
            test_imgs.append(('%s_%02d')%(u_id,dir))
    with open(os.path.join(TEXT_PATH,'train_imgs.txt'),'w') as writer:
        lines = '\n'.join(train_imgs)
        writer.writelines(lines)
    with open(os.path.join(TEXT_PATH,'test_imgs.txt'),'w') as writer:
        lines = '\n'.join(test_imgs)
        writer.writelines(lines)








