# -*- coding:utf8 -*-

import os
from PIL import Image
from shutil import copyfile, copytree, rmtree, move

PATH_DATASET = './car-dataset' # Source folder to process.
PATH_NEW_DATASET = './car-reid-dataset' # Processed output folder.
PATH_ALL_IMAGES = PATH_NEW_DATASET + '/all_images'
PATH_TRAIN = PATH_NEW_DATASET + '/train'
PATH_TEST = PATH_NEW_DATASET + '/test'

# Directory creation helper.
def mymkdir(path):
    path = path.strip() # Remove leading and trailing spaces.
    path = path.rstrip("\\") # Remove trailing backslash.
    isExists = os.path.exists(path) # Check whether the path exists.
    if not isExists:
        os.makedirs(path) # Create the directory if it does not exist.
        print(path + ' created successfully')
        return True
    else:
        # Do not create the directory if it already exists.
        print(path + ' directory already exists')
        return False

class BatchRename():
    '''
    Batch rename image files in a folder.
    '''

    def __init__(self):
        self.path = PATH_DATASET # Folder containing files to rename.

    # Resize images.
    def resize(self):
        for aroot, dirs, files in os.walk(self.path):
            # aroot is each subdirectory under self.path, including self.path.
            # dirs is the list of folders under self.path.
            filelist = files  # This is the file list for the current path.
            # print('list', list)

            # filelist = os.listdir(self.path) # Get file paths.
            total_num = len(filelist)  # Number of files.

            for item in filelist:
                if item.endswith('.jpg'):  # Source images are jpg; adjust this for png or other formats as needed.
                    src = os.path.join(os.path.abspath(aroot), item)

                    # Resize images to 128 width x 256 height.
                    im = Image.open(src)
                    out = im.resize((128, 256), Image.ANTIALIAS)  # resize image with high-quality
                    out.save(src)  # Save back to the original path.

    def rename(self):

        for aroot, dirs, files in os.walk(self.path):
            # aroot is each subdirectory under self.path, including self.path.
            # dirs is the list of folders under self.path.
            filelist = files  # This is the file list for the current path.
            # print('list', list)

            # filelist = os.listdir(self.path) # Get file paths.
            total_num = len(filelist)  # Number of files.

            i = 1  # File numbering starts from 1.
            for item in filelist:
                if item.endswith('.jpg'):  # Source images are jpg; adjust this for png or other formats as needed.
                    src = os.path.join(os.path.abspath(aroot), item)

                    # Create an image directory based on the image name.
                    dirname = str(item.split('_')[0])
                    # Create a directory for the same vehicle.
                    #new_dir = os.path.join(self.path, '..', 'bbox_all', dirname)
                    new_dir = os.path.join(PATH_ALL_IMAGES, dirname)
                    if not os.path.isdir(new_dir):
                        mymkdir(new_dir)

                    # Get the number of images in new_dir.
                    num_pic = len(os.listdir(new_dir))

                    dst = os.path.join(os.path.abspath(new_dir),
                                       dirname + 'C1T0001F' + str(num_pic + 1) + '.jpg')
                    # The processed format is also jpg. This can be changed to png if needed.
                    # C1T0001F follows the mars.py filename format: camera ID and track index.
                    # This alternative names files like 0000000.jpg; customize the format as needed.
                    # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
                    try:
                        copyfile(src, dst) #os.rename(src, dst)
                        print ('converting %s to %s ...' % (src, dst))
                        i = i + 1
                    except:
                        continue
            print ('total %d to rename & converted %d jpgs' % (total_num, i))
            
    def split(self):
        #---------------------------------------
        #train_test
        images_path = PATH_ALL_IMAGES
        train_save_path = PATH_TRAIN
        test_save_path = PATH_TEST
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)
            os.mkdir(test_save_path)
        
        for _, dirs, _ in os.walk(images_path, topdown=True):
            for i, dir in enumerate(dirs):
                for root, _, files in os.walk(images_path + '/' + dir, topdown=True):
                    for j, file in enumerate(files):
                        if(j==0): # Test dataset; the first image of each vehicle.
                            print("index: %s  folder: %s  image: %s assigned to test set" % (i + 1, root, file))
                            src_path = root + '/' + file
                            dst_dir = test_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
                        else:
                            src_path = root + '/' + file
                            dst_dir = train_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
        rmtree(PATH_ALL_IMAGES)
        
if __name__ == '__main__':
    demo = BatchRename()
    demo.resize()
    demo.rename()
    demo.split()


