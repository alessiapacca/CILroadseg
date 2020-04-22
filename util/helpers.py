import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import matplotlib.image as mpimg
import math

# Loads image and mask given the image path.
# Parameter: image_path
# Returns  : image and mask as a tuple of Pillow.Image
def load_image_and_mask(image_path):
    file_name = os.path.split(image_path)[1]
    img = Image.open(str(image_path))
    mask_path = "training" + os.path.sep + "training" + \
        os.path.sep + "groundtruth" + os.path.sep + file_name
    mask = Image.open(str(mask_path))
    return (img, mask)

# Loads image given the image path.
# Parameter: image_path
# Returns  : image as a Pillow.Image
def load_image(infilename):
    return mpimg.imread(infilename)

# Pads Image and Mask with zeroes
# Parameter: image and mask
# Returns  : image and mask as a tuple
def pad_image_and_mask(img, mask, new_size):
    old_width, old_height = img.size
    new_image = img.resize((new_size, new_size), Image.LANCZOS)
    new_mask = mask.resize((new_size, new_size), Image.LANCZOS)

    new_image = Image.new("RGB", (new_size, new_size))
    new_image.paste(img, ((new_size-old_width)//2, (new_size-old_height)//2))

    new_mask = Image.new("RGB", (new_size, new_size))
    new_mask.paste(mask, ((new_size-old_width)//2, (new_size-old_height)//2))
    return (new_image, new_mask)

# Pads Image with zeroes
# Parameter: image
# Returns  : image padded
def pad_image(img, new_size):
    return img.resize((new_size, new_size), Image.LANCZOS)

# Flips image and mask with respect to vertical axis
# Parameter: image and mask
# Returns  : image and mask as a tuple
def flip_image_and_mask(img, mask):
    flipped_img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    flipped_mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
    return (flipped_img, flipped_mask)

# Flips image with respect to vertical axis
# Parameter: image
# Returns  : image 
def flip_image(img):
    return img.transpose(method=Image.FLIP_LEFT_RIGHT)

# Displays image and respective mask in matplotlib window
# Parameters : Image and mask as Pillow.Image
def view_image_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    a.set_title("original image")
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    a.set_title("mask")
    plt.show()

# Displays a list of images and respective masks in matplotlib window
# Parameters : List of Images and masks as Pillow.Image
def view_images(images, masks):
    cols = 2
    n_images = len(images)
    fig = plt.figure()
    for n in range(n_images):
        fig.add_subplot(np.ceil(n_images * 2/float(cols)), cols, 2 * n + 1)
        plt.imshow(images[n])
        fig.add_subplot(np.ceil(n_images * 2/float(cols)), cols, 2 * n + 2)
        plt.imshow(masks[n])
    plt.show()

# Displays a single image as PDF
# Parameter : image to display
def view_image(img):
    img.show()

# Zoom image and mask
# Parameter: image, mask, coordinates of center of zoom and zoom value
# Returns  : image and mask as a tuple
def zoom_img_and_mask(img, mask, x, y, zoom):
    assert(img.size == mask.size)
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    mask = mask.crop((x - w / zoom2, y - h / zoom2,
                      x + w / zoom2, y + h / zoom2))
    img = img.resize((w, h), Image.LANCZOS)
    mask = mask.resize((w, h), Image.LANCZOS)
    return (img, mask)

# Zoom image
# Parameter: image, coordinates of center of zoom and zoom value
# Returns  : image and mask as a tuple
def zoom_img(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)

# Rotates an image and mask by a given angle
# Parameters: Image and mask to rotate, angle of rotation (in radians)
# Returns   : Rotated image and mask as a tuple
def rotate_img_and_mask(img, mask, angle):
    img = img.rotate(angle)
    mask = mask.rotate(angle)
    return (img, mask)

# Rotates an image by a given angle
# Parameters: Image to rotate, angle of rotation (in radians)
# Returns   : Image
def rotate_img(img, angle):
    return img.rotate(angle)

# Crops image and mask to create a list of squares resized
# Parameters: image and mask to crop, height and width of new cropped images
# Returns   : list of new cropped images and masks
def crop_image_and_mask(img, mask, height, width):
    list_img = []
    list_mask = []
    imgwidth, imgheight = img.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = img.crop(box)
            a = a.resize((imgwidth, imgheight), Image.LANCZOS)
            list_img.append(a)
            a = mask.crop(box)
            a = a.resize((imgwidth, imgheight), Image.LANCZOS)
            list_mask.append(a)
    return (list_img, list_mask)

# Saves images and masks inside lists into a new directory.
# Requires a "saved/images" and a "saved/groundtruth" folder.
# Parameters : images and masks list to save
def save_images(imgs, masks):
    n_images = len(imgs)
    for i in range(n_images):
        imgs[i].save("saved" + os.path.sep + "images" + os.path.sep + str(i) + ".png","PNG")
        masks[i].save("saved" + os.path.sep + "groundtruth" + os.path.sep + str(i) + ".png","PNG")

# Crop an image into patches. This method expects grey scale images. 
# Parameters : Image to be cropped, width, height and stride of the patch. Each patch of w*h will be cropped out. 
# Stride determines the shift of the window. 
# Returns : A list containing the cropped files. 

def img_crop_gt(im, w, h, stride):
    """ Crop an image into patches (this method is intended for ground truth images). """
    assert len(im.shape) == 2, 'Expected greyscale image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(0,imgheight,stride):
        for j in range(0,imgwidth,stride):
            im_patch = im[j:j+w, i:i+h]
            list_patches.append(im_patch)
    return list_patches
 
# Crop an image into patches. 
# Parameters : Image to be cropped, width, height and stride, and padding required.Each patch of w*h will be cropped out. 
# Stride determines the shift of the window. 
# Returns : A list containing the cropped files. 

def img_crop(im, w, h, stride, padding):
    """ Crop an image into patches, taking into account mirror boundary conditions. """
    assert len(im.shape) == 3, 'Expected RGB image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    for i in range(padding,imgheight+padding,stride):
        for j in range(padding,imgwidth+padding,stride):
            im_patch = im[j-padding:j+w+padding, i-padding:i+h+padding, :]
            list_patches.append(im_patch)
    return list_patches

# Create 16*16 patches for ground truth images. 
def create_patches_gt(X, patch_size, stride):
    img_patches = np.asarray([img_crop_gt(X[i], patch_size, patch_size, stride) for i in range(X.shape[0])])
    # Linearize list
    img_patches = img_patches.reshape((-1, img_patches.shape[2], img_patches.shape[3]))
    return img_patches

# Get Percentage of correct answers
def get_classification_results(y, y_test):
    """
    Get the ratio of correct answers.
    """
    y = y.reshape(-1) # Linearize
    y_test = y_test.reshape(-1) # Linearize
    diff = y - y_test
    correct = np.sum(diff == 0)
    return correct / y_test.size


def patchify(Y, patch_size):
    patches = (np.mean(create_patches_gt(Y, patch_size, patch_size), axis=(1, 2)) > 0.25) * 1
    return patches.reshape(Y.shape[0], -1)

# assumes one classification per patch
def recompose(Y, num_of_img, img_size, patch_size):
    Y = Y.reshape((num_of_img, math.ceil(img_size[0] / patch_size), math.ceil(img_size[1] / patch_size)))
    Y = np.transpose(Y, axes=[0, 2, 1])

    Y = np.repeat(Y, patch_size, axis=1)
    Y = np.repeat(Y, patch_size, axis=2)

    return Y[:, 0:img_size[0], 0:img_size[1]]




















