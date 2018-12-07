#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 0. libraries -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

from math import *
from Functions import *
from PIL import Image
from scipy.misc import imsave
from os import *

#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Initial settings. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

# some of the varias in comments  are for further used
chdir(r"D:\PRP_2018_fall\code1")
# !! change to the adrress of working file
shape_factor = 16 # !! root factor of pic reading
window_shape = (shape_factor , shape_factor)    # !! Patches' shape
#sigma = 10                 # Noise standard dev.
resize_shape = (window_shape[0]*window_shape[0], window_shape[0]*window_shape[0])  # Resized image's shape
#step = int(shape_factor/2)                # Patches' step
#ratio = 1             # Ratio for the dictionary (training set).
#ksvd_iter = 5              # Number of iterations for the K-SVD
dic_num = 256
#-------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- 2. Image import. ----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

name = r'new2.jpg'
read_add = r''
original_image = np.asarray(Image.open(read_add + name).convert('L').resize(resize_shape))
learning_image = np.asarray(Image.open(read_add + name).convert('L').resize(resize_shape))

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- 3. Image processing. --------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
face = original_image
# make face as the image imported


# Convert from uint8 representation with values between 0 and 255 to
# a floating point representation with values between 0 and 1.
face = face / 255.0


# downsample for higher speed
# make every 4 pixels into one
# 1 1   ->    4
# 1 1
#old_face = face
#face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
#face = face / 4.0
height, width = face.shape



#-------------------# we don't need distort------------------------
# Distort the right half of the image
print('Distorting image...')
distorted = face.copy()
#.copy of np is a deep copy
# distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)
#==================================================================


#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 4. generate Dictionary. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
# Extract all patches from theimage
print('Extracting reference patches...')
t0 = time()
patch_size = (16, 16)
data = extract_patches_2d(distorted[:, :], patch_size)
#data = extract_patches_2d(distorted[:, :width // 2], patch_size)
data = data.reshape(data.shape[0], -1)
print(data.shape)

data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))
print('Learning the dictionary...')

t0 = time()
dico = MiniBatchDictionaryLearning(n_components=dic_num, alpha=1, n_iter=200)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:dic_num]):
    c=comp.reshape(patch_size)
    break

for i, comp in enumerate(V[:dic_num]):#this is the data interface
    plt.subplot( ceil(dic_num**0.5), ceil(dic_num**0.5), i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    if(i!=0):
        c = np.concatenate((c, comp.reshape(patch_size)), axis = None)
    plt.xticks(())
    plt.yticks(())

plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)#left, right, bottom, top, wspace, hspace

#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 4. generate phi and reconstruct. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
transform_algorithms =('Orthogonal Matching Pursuit\n1 atom', 'omp',{'transform_n_nonzero_coefs': 1})
data = extract_patches_2d(face, patch_size)
data = data.reshape(data.shape[0], -1)
print(data.shape)
code = dico.transform(data)
patches = np.dot(code, V)
patches = patches.reshape(len(data), *patch_size)
reconstructions= face.copy()
reconstructions[:,:] = reconstruct_from_patches_2d(patches, (height, width))
reconstructions = reconstructions*255.0
plt.figure('original_image & reconstructions')
plt.subplot(1,2,1)
plt.imshow(face,cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(reconstructions,cmap=plt.cm.gray)
plt.figure(figsize=(4.2, 4))
plt.imshow(c.reshape(resize_shape), cmap=plt.cm.gray_r,
               interpolation='nearest')
sp.misc.imsave('outfile.jpg', c)
plt.show()
