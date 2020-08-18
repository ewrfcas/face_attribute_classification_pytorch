import os
import argparse
from utils.utils import check_args, torch_show_all_params, torch_init_model
from utils.pytorch_optimization import get_linear_schedule_with_warmup
from data_loader import create_dataloader
from model import AttClsModel
import torch
import numpy as np
import random
from tqdm import tqdm
from utils.logger import setup_logger


def load_data(path):
    label_dict = {}
    img_list = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            d = line.split()
            img_list.append(d[0])
            label_dict[d[0]] = []
            for label in d[1:]:
                label_dict[d[0]].append(int(float(label) * 0.5 + 0.5))
    return img_list, label_dict


def load_atts(path):
    with open(path, 'r') as f:
        for line in f:
            atts = line.split()
    return atts


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='resnet50')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--fix_epoch', type=float, default=0.4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=int, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--adaptive_weights', type=bool, default=True)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=556)

    # dir
    parser.add_argument('--input_path', type=str,
                        default='../img_align_celeba')
    parser.add_argument('--train_path', type=str,
                        default='data_list/align_train/align_train_attr.txt')
    parser.add_argument('--dev_path', type=str,
                        default='data_list/align_val/align_val_attr.txt')
    parser.add_argument('--test_path', type=str,
                        default='data_list/align_test/align_test_attr.txt')
    parser.add_argument('--att_path', type=str,
                        default='data_list/att_map.txt')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/FAC_resnet50_AW_V1')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    logger = setup_logger(args.checkpoint_dir, logfile_name=args.log_file, logger_name='att_cls')

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12340'
        from torch.distributed import init_process_group
        init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
    check_args(args, rank=local_rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load dataset
    train_list, train_label_dict = load_data(args.train_path)
    dev_list, dev_label_dict = load_data(args.dev_path)
    test_list, test_label_dict = load_data(args.test_path)

    if n_gpu > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(np.arange(len(train_list)), num_replicas=n_gpu,
                                           rank=local_rank, shuffle=True)
        # 仅用于调整lambda
        auxiliary_sampler = DistributedSampler(np.arange(len(dev_list)), num_replicas=n_gpu,
                                               rank=local_rank, shuffle=True)
    else:
        from torch.utils.data.sampler import RandomSampler
        train_sampler = RandomSampler(np.arange(len(train_list)))
        auxiliary_sampler = RandomSampler(np.arange(len(dev_list)))
    train_loader = create_dataloader(args, args.batch_size, train_list, args.input_path, train_label_dict,
                                     is_train=True, n_threads=n_gpu, sampler=train_sampler)
    auxiliary_loader = create_dataloader(args, args.batch_size, dev_list, args.input_path, dev_label_dict,
                                         is_train=True, n_threads=n_gpu, sampler=auxiliary_sampler)
    dev_loader = create_dataloader(args, args.batch_size * n_gpu, dev_list, args.input_path,
                                   dev_label_dict, is_train=False, n_threads=n_gpu)
    test_loader = create_dataloader(args, args.batch_size * n_gpu, test_list, args.input_path,
                                    test_label_dict, is_train=False, n_threads=n_gpu)
    att_map = load_atts(args.att_path)

    # create model
    device = torch.device("cuda")
    model = AttClsModel(args, device=device)
    model.to(device)
    if n_gpu > 1:
        if not args.float16:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            from apex.parallel import convert_syncbn_model
            model = convert_syncbn_model(model)

    if local_rank == 0:
        logger.info('Parameters: ' + str(torch_show_all_params(model)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    total_steps = int(args.epoch * (len(train_list) / args.batch_size / n_gpu))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters,
                                                fix_steps=int(args.fix_epoch * total_steps),
                                                num_training_steps=total_steps)
    if args.float16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if n_gpu > 1:
        if not args.float16:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[local_rank],
                                                              output_device=local_rank,
                                                              find_unused_parameters=True)
        else:
            from apex.parallel import DistributedDataParallel
            model = DistributedDataParallel(model)

    current_step = 0
    start_epoch = 0

    # training
    model.train()
    show_loss = 0

    best_mean_acc = 0
    if local_rank == 0:
        logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, args.epoch):
        if n_gpu > 1:
            train_sampler.set_epoch(epoch)
            auxiliary_sampler.set_epoch(epoch)
        aux_iter = iter(auxiliary_loader)
        for train_data in train_loader:
            train_img, train_labels = train_data

            try:
                aux_data = next(aux_iter)
            except StopIteration:
                aux_iter = iter(auxiliary_loader)
                aux_data = next(aux_iter)

            aux_img, aux_labels = aux_data
            train_img = train_img.to(device=device)
            train_labels = train_labels.to(device=device)
            loss = model(train_img, train_labels)

            show_loss += loss.detach().item()
            if args.float16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            # update lambda weights
            if args.adaptive_weights:
                model.eval()
                with torch.no_grad():
                    aux_img = aux_img.to(device=device)
                    aux_labels = aux_labels.to(device=device)
                    model.adaptive_update_weights(aux_img, aux_labels, current_step)
                model.train()

            current_step += 1

            # log
            if current_step % args.print_freq == 0 and local_rank == 0:
                message = '[epoch:{0}/{1}, steps:{2}/{3}, lr:{4:.3e}, loss:{5:.5}] '.format(epoch, args.epoch,
                                                                                            current_step, total_steps,
                                                                                            scheduler.get_lr()[0],
                                                                                            show_loss / args.print_freq)
                show_loss = 0
                logger.info(message)

        # validation
        if local_rank == 0:
            model.eval()
            att_wrong = np.zeros(40)
            with torch.no_grad():
                for val_data in tqdm(dev_loader):
                    val_img, val_labels = val_data
                    logits = model(val_img.to(device))
                    val_labels = val_labels.numpy()
                    logits = logits.detach().cpu().numpy()
                    preds = np.zeros_like(logits)
                    preds[logits > 0] = 1
                    diff = np.abs(preds - val_labels)  # [bs, 40]
                    diff = np.sum(diff, axis=0)  # [40,]
                    att_wrong += diff

            att_wrong /= len(dev_list)
            att_acc = 1 - att_wrong
            val_mean_acc = np.mean(att_acc)
            message = '[epoch:{0} end, mean_acc:{1:.5}]'.format(epoch, val_mean_acc)
            logger.info(message)
            att_scores = {}
            for j in range(len(att_map)):
                att_scores[att_map[j]] = att_acc[j]
            logger.info(str(att_scores))

            # save models
            print('Saving models and training states.')
            if val_mean_acc > best_mean_acc:
                best_mean_acc = val_mean_acc
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(),
                           os.path.join(args.checkpoint_dir, 'best_model.pth'))
            else:
                torch.save(model.state_dict(),
                           os.path.join(args.checkpoint_dir, 'best_model.pth'))
            model.train()
            # validating end

    if local_rank == 0:
        logger.info('End of training.')
        logger.info('Max eval mean acc: {}'.format(best_mean_acc))

        torch_init_model(model, os.path.join(args.checkpoint_dir, 'best_model.pth'))

        # testing
        model.eval()
        att_wrong = np.zeros(40)
        with torch.no_grad():
            for test_data in tqdm(test_loader):
                test_img, test_labels = test_data
                logits = model(test_img.to(device))
                test_labels = test_labels.numpy()
                logits = logits.detach().cpu().numpy()
                preds = np.zeros_like(logits)
                preds[logits > 0] = 1
                diff = np.abs(preds - test_labels)  # [bs, 40]
                diff = np.sum(diff, axis=0)  # [40,]
                att_wrong += diff

        att_wrong /= len(test_list)
        att_acc = 1 - att_wrong
        test_mean_acc = np.mean(att_acc)
        message = '[test mean_acc:{:.5}]'.format(test_mean_acc)
        logger.info(message)
        att_scores = {}
        for j in range(len(att_map)):
            att_scores[att_map[j]] = att_acc[j]
        logger.info(str(att_scores))


if __name__ == '__main__':
    main()
