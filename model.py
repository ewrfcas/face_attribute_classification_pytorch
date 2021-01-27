import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models


class AttClsModel(nn.Module):
    def __init__(self, opt, device):
        super(AttClsModel, self).__init__()
        if opt.model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            hidden_size = 2048
        elif opt.model_type == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            hidden_size = 512
        elif opt.model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            hidden_size = 512
        else:
            raise NotImplementedError
        self.lambdas = torch.ones((40,), device=device)
        if hasattr(opt, 'k'):
            self.k = opt.k
        else:
            self.k = -1
        self.val_loss = []  # max_len == 2*k
        self.fc = nn.Linear(hidden_size, 40)
        self.dropout = nn.Dropout(0.5)

    def backbone_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        return x

    def forward(self, input, labels=None):
        x = self.backbone_forward(input)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if labels is None:
            return x
        else:
            # [bs, 40]
            cls_loss = nn.BCEWithLogitsLoss(reduction='none')(input=x, target=labels)
            cls_loss = torch.sum(cls_loss * self.lambdas.unsqueeze(0), dim=1)
            cls_loss = torch.mean(cls_loss)
            return cls_loss

    # these code should be run after eval()
    def adaptive_update_weights(self, val_input, val_labels, step, eps=1e-8):
        x = self.backbone_forward(val_input)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        val_loss = nn.BCEWithLogitsLoss(reduction='none')(input=x, target=val_labels)
        val_loss = torch.mean(val_loss, dim=0)  # [40,]
        self.val_loss.append(val_loss)
        self.val_loss = self.val_loss[-(2 * self.k):]
        if step % self.k == 0 and len(self.val_loss) == 2 * self.k:
            # [2k, 40]
            val_loss_list = torch.cat([vl.unsqueeze(0) for vl in self.val_loss], dim=0)
            pre_mean = torch.mean(val_loss_list[:self.k, :], dim=0)  # [40,]
            cur_mean = torch.mean(val_loss_list[self.k:, :], dim=0)
            trend = torch.abs(cur_mean - pre_mean) / (cur_mean + eps)
            norm_trend = trend / (torch.mean(trend) + eps)
            norm_loss = cur_mean / (torch.mean(cur_mean) + eps)
            self.lambdas = norm_trend * norm_loss
            self.lambdas = self.lambdas / (torch.mean(self.lambdas) + eps)
