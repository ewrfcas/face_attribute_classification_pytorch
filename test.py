import os
import argparse
from utils.utils import torch_init_model
from model import AttClsModel
import torch
from PIL import Image
import torchvision.transforms as transforms


def load_atts(path):
    with open(path, 'r') as f:
        for line in f:
            atts = line.split()
    return atts


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='7')
    parser.add_argument('--model_type', type=str, default='resnet50')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--float16', type=bool, default=False)

    # dir
    parser.add_argument('--img_path', type=str,
                        default='../img_align_celeba')
    parser.add_argument('--att_path', type=str,
                        default='data_list/att_map.txt')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/FAC_resnet50_AW_V1')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # load dataset
    img = Image.open(args.img_path)
    transform_list = []
    transform_list.append(transforms.Resize((args.img_size, args.img_size)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    img2tensor = transforms.Compose(transform_list)
    img = img2tensor(img)

    att_map = load_atts(args.att_path)

    # create model
    device = torch.device("cuda")
    model = AttClsModel(args, device=device)
    model.to(device)
    torch_init_model(model, os.path.join(args.checkpoint_dir, 'best_model.pth'))

    # testing
    model.eval()
    att_preds = {}
    with torch.no_grad():
        logits = model(img.to(device).unsqueeze(0))
        sigmoid_probs = torch.sigmoid(logits)[0]
        sigmoid_probs = sigmoid_probs.detach().cpu().numpy()

    for j in range(len(att_map)):
        att_preds[att_map[j]] = float(sigmoid_probs[j])

    import json
    print(json.dumps(att_preds, indent=2))


if __name__ == '__main__':
    main()
