from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch


def create_dataloader(args, batch_size, img_list, input_path, label_dict,
                      n_threads=1, is_train=True, sampler=None):
    return DataLoader(
        celeba_dataloader(args, img_list, input_path, label_dict, is_train),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        sampler=sampler,
        drop_last=False
    )


class celeba_dataloader(Dataset):
    def __init__(self, args, img_list, input_path, label_dict, is_train):
        super(celeba_dataloader, self).__init__()
        self.args = args
        self.img_list = img_list
        self.input_path = input_path
        self.is_train = is_train
        self.label_dict = label_dict
        self.img_trans = self.img_transformer()

    def __len__(self):
        return len(self.img_list)

    def img_transformer(self):
        transform_list = []
        if self.is_train:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.Resize((self.args.img_size, self.args.img_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        img2tensor = transforms.Compose(transform_list)

        return img2tensor

    def __getitem__(self, index):
        file_name = self.img_list[index]
        input_path = os.path.join(self.input_path, file_name)
        img = Image.open(input_path)
        img = img.crop((0, 20, 178, 198))
        img = self.img_trans(img)
        labels = torch.tensor(self.label_dict[file_name]).to(dtype=torch.float32)
        return img, labels
