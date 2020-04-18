import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

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
def load_image(path):
    return Image.open(str(path))

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
        print(n)
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