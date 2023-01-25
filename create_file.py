import numpy as np
# import torch
import matplotlib.pyplot as plt
import h5py

def createShape(shape_size, shape_type):
    h, w = shape_size
    img = np.zeros(shape_size)
    if shape_type == 0:
        img += 1
    elif shape_type == 1:
        vertex_coord = w // 2
        for row in range(h):
            ratio = (row + 1) / h
            start_ndx = int(vertex_coord * (1 - ratio))
            end_ndx = vertex_coord + int(vertex_coord * ratio)
            img[row, start_ndx:end_ndx] = 1
    elif shape_type == 2:
        center_h = h // 2
        center_w = w // 2
        radius = min(center_h, center_w, h - center_h - 1, w - center_w - 1)

        y, x = np.ogrid[:h, :w]
        dist_from_center = (x - center_w)**2 + (y - center_h)**2
        mask = dist_from_center <= radius**2
        img[mask] = 1

    shape_mask = img == 1

    return img, shape_mask

def createImage(image_size, colour_mode):
    h, w = image_size
    shape_choice = np.random.permutation(4)

    shape_h, shape_w = [h // 2, w // 2]
    shape_size = [shape_h, shape_w]
    shape1, shape_mask1 = createShape(shape_size, shape_choice[0])
    shape2, shape_mask2 = createShape(shape_size, shape_choice[1])
    shape3, shape_mask3 = createShape(shape_size, shape_choice[2])
    shape4, shape_mask4 = createShape(shape_size, shape_choice[3])

    offset1 = np.random.randint(0, [h // 8, w // 8])
    offset2 = np.random.randint(0, [h // 8, w // 8])
    offset2[1] = offset2[1] + (w * 3) // 8
    offset3 = np.random.randint(0, [h // 8, w // 8])
    offset3[0] = offset3[0] + (h * 3) // 8
    offset4 = np.random.randint(0, [h // 8, w // 8])
    offset4[0] = offset4[0] + (h * 3) // 8
    offset4[1] = offset4[1] + (w * 3) // 8 

    mask = np.zeros([4, h, w]) == 1
    mask[0, offset1[0]: offset1[0] + shape_h, offset1[1]: offset1[1] + shape_w] = shape_mask1
    mask[1, offset2[0]: offset2[0] + shape_h, offset2[1]: offset2[1] + shape_w] = shape_mask2
    mask[2, offset3[0]: offset3[0] + shape_h, offset3[1]: offset3[1] + shape_w] = shape_mask3
    mask[3, offset4[0]: offset4[0] + shape_h, offset4[1]: offset4[1] + shape_w] = shape_mask4

    square_ndx = np.where(shape_choice == 0)[0][0]
    triangle_ndx = np.where(shape_choice == 1)[0][0]
    circle_ndx = np.where(shape_choice == 2)[0][0]

    square_mask = mask[square_ndx, :, :]
    triangle_mask = mask[triangle_ndx, :, :]
    circle_mask = mask[circle_ndx, :, :]

    colour_choice = np.random.permutation(4)
    mask = mask[colour_choice]

    if colour_mode == 'grey':
        img = np.zeros(image_size)
        mask = np.amax(mask, axis=0)
        img[mask] = 1
    elif colour_mode == 'rgb':
        empty_ndx = np.argmax(shape_choice)
        empty_colour = np.where(colour_choice == empty_ndx)[0][0]
        img = np.zeros([4, h, w])
        img[mask] = 1
        img = np.delete(img, empty_colour, 0)
        img = np.transpose(img, (1, 2, 0))

    return img, square_mask, triangle_mask, circle_mask

def generateAndSaveFile(img_dir, image_size, colour_mode):
    img, square_mask, triangle_mask, circle_mask = createImage(image_size, colour_mode)

    f = h5py.File(img_dir, 'w')
    f.create_dataset('img', data=img, chunks=True)
    f.create_dataset('square', data=square_mask, chunks=True)
    f.create_dataset('triangle', data=triangle_mask, chunks=True)
    f.create_dataset('square', data=circle_mask, chunks=True)
    
# shape = createShape([100,100], 3)
# img = torch.zeros([200, 200]).numpy()
# img[:100,:100] = shape
img, s_mask, t_mask, c_mask = createImage([100, 100], 'rgb')
fig = plt.figure(figsize=(2,2))
fig.add_subplot(2, 2, 1)
plt.imshow(img)
fig.add_subplot(2, 2, 2)
plt.imshow(s_mask)
fig.add_subplot(2, 2, 3)
plt.imshow(t_mask)
fig.add_subplot(2, 2, 4)
plt.imshow(c_mask)

# imgplot = plt.imshow(img)
plt.show()