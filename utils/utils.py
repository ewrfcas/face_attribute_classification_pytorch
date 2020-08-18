import os
import cv2
import numpy as np
import torch
from PIL import Image


def check_args(args, rank=0):
    args.setting_file = os.path.join(args.checkpoint_dir, args.setting_file)
    if os.path.exists(args.setting_file) and rank == 0:
        os.remove(args.setting_file)
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(args.setting_file, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')


# utils
def tensor2im(input_image, imtype=np.uint8, show_size=128):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.detach().cpu().float().numpy()
    im = []
    for i in range(image_numpy.shape[0]):
        im.append(
            np.array(numpy2im(image_numpy[i], imtype).resize((show_size, show_size), Image.ANTIALIAS)))
    return np.array(im)


def numpy2im(image_numpy, imtype=np.uint8):
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) / 2. + 0.5) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = image_numpy.astype(imtype)
    im = Image.fromarray(image_numpy)
    return im


def display_online_results(visuals, steps, vis_saved_dir, show_size=128, channel=3, prefix_zero=4):
    images = []
    labels = []
    for label, image in visuals.items():
        image_numpy = tensor2im(image, show_size=show_size)  # [batch, show_size, show_size, channel]
        image_numpy = np.reshape(image_numpy, (-1, show_size, channel))
        images.append(image_numpy)
        labels.append(label)
    save_images = np.array(images)  # [label_num, show_size*batch, show_size, channel]
    save_images = np.transpose(save_images, [1, 0, 2, 3])
    # [batch*show_size, label_num*show_size, channel]
    save_images = np.reshape(save_images, (save_images.shape[0], -1, channel))
    title_img = get_title(labels, show_size)
    save_images = np.concatenate([title_img, save_images], axis=0)
    prefix_steps = str(steps)
    while len(prefix_steps) < prefix_zero:
        prefix_steps = '0' + prefix_steps
    save_image(save_images, os.path.join(vis_saved_dir, 'display_' + prefix_steps + '.jpg'))


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def get_title(labels, show_size=128):
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_img = []
    for label in labels:
        x = np.ones((40, show_size, 3)) * 255.0
        textsize = cv2.getTextSize(label, font, 0.5, 2)[0]
        x = cv2.putText(x, label, ((x.shape[1] - textsize[0]) // 2, x.shape[0] // 2), font, 0.5, (0, 0, 0), 1)
        title_img.append(x)

    title_img = np.array(title_img)
    title_img = np.transpose(title_img, [1, 0, 2, 3])
    title_img = np.reshape(title_img, [title_img.shape[0], -1, 3])
    title_img = title_img.astype(np.uint8)

    return title_img


def torch_show_all_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def torch_init_model(model, init_checkpoint):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))
