from ast import mod
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from typing import Dict, List, Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
from .singe_prototype import Singe_prototype
# import ipdb
import os

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),  # 卷积块 卷积加批量标准化 激活
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, feature1, feature2):
        # 计算均方误差
        mse_loss = nn.MSELoss()(feature1, feature2)
        return mse_loss

class invariant(nn.Module):
    def __init__(self):
        super(invariant, self).__init__()
        self.di = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1024),
            )
    def forward(self, x):
        x = self.di(x)
        return x

class specific(nn.Module):
    def __init__(self):
        super(specific, self).__init__()
        self.di = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(1024),
            )
    def forward(self, x):
        x = self.di(x)
        return x

class PrototypeClusteringModule(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(PrototypeClusteringModule, self).__init__()

        self.K = num_classes
        self.C = in_channels

        # 1. Soft Assignment 模块 (ConvSoft): C -> K
        self.conv_soft = nn.Conv2d(in_channels, self.K, kernel_size=1)

        # 可学习的聚类中心 sp (K, C)
        self.sp = nn.Parameter(torch.randn(self.K, self.C))

        # 2. 加权残差处理后的全连接层 (Linear): K*C -> K*C
        self.fc = nn.Linear(self.K * self.C, self.K * self.C)

        # 3. 原型增强融合卷积 (Conv): 2*C -> C
        self.conv_fuse = nn.Conv2d(self.C * 2, self.C, kernel_size=1)

        # LayerNorm 保持训练稳定，沿通道 C 维度
        self.ln = nn.LayerNorm(self.C)

    def forward(self, F):
        # F: (N, C, H, W)
        N, C, H, W = F.shape

        # ----------------- Step 1: 计算软分配概率 θ (N, K, H, W) -----------------
        F_norm = F / (torch.norm(F, p=2, dim=1, keepdim=True) + 1e-6)  # L2 归一化
        theta = self.conv_soft(F_norm)
        theta = torch.nn.functional.softmax(theta, dim=1)

        # ----------------- Step 2: 计算加权残差 F1_p (N*H*W, K, C) -----------------

        F_reshaped = F.permute(0, 2, 3, 1).reshape(N * H * W, C)  # (N*H*W, C)

        # 扩展 F 和 sp 以计算残差 (像素特征 - 类别原型)
        sp_expanded = self.sp.unsqueeze(0).expand(N * H * W, self.K, C)
        F_expanded = F_reshaped.unsqueeze(1).expand_as(sp_expanded)
        F_res = F_expanded - sp_expanded

        # 扩展 theta (N*H*W, K) 并加权残差
        theta_reshaped = theta.permute(0, 2, 3, 1).reshape(N * H * W, self.K)
        F1_p = F_res * theta_reshaped.unsqueeze(2)

        # ----------------- Step 3: L2 sum F1_p 得到 Fp2 (N, K, C) -----------------

        # Sum along pixel dimension
        F1_p_summed = F1_p.reshape(N, H * W, self.K, C).sum(dim=1)

        # 2-norm normalization (F2_p)
        F2_p = F1_p_summed / (torch.norm(F1_p_summed, p=2, dim=2, keepdim=True) + 1e-6)
        F2_p_reshaped = F2_p.reshape(N, self.K * self.C)  # (N, K*C)

        # ----------------- Step 4: 全连接层得到 Fp3 (N, C, H, W) -----------------

        F3_p_linear = self.fc(F2_p_reshaped)
        F3_p_reshaped = F3_p_linear.reshape(N, self.K, self.C)

        # 关键：将 (N, K, C) 广播回空间维度 (N, C, H, W)
        F3_p_avg = F3_p_reshaped.mean(dim=1)  # (N, C)
        F3_p = F3_p_avg.unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W)

        # ----------------- Step 5: 融合得到 Fp (N, C, H, W) -----------------

        F_cat = torch.cat((F, F3_p), dim=1)  # (N, 2*C, H, W)
        Fp_enhanced = self.conv_fuse(F_cat)

        # 使用残差连接和 LayerNorm 增强
        Fp = F + Fp_enhanced
        # LayerNorm 适用于 (N, H, W, C)，再转回来
        Fp = self.ln(Fp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 只返回原型增强特征 Fp
        return Fp

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
        self.pro = Singe_prototype(1024, 7)
        self.pro2 = Singe_prototype(1024, 7)
        self.di = invariant()
        self.ds = specific()
        self.conv_out = convblock(2 * 1024, 1024, 3, 1, 1)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        clip_images = [T.functional.normalize(ci.flip(0) / 255, mean, std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i for i in clip_images])
        return clip_images

    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def generate_colors(self, N):
        import colorsys
        '''
            Generate random colors.
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        '''
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
        perm = np.arange(7)
        colors = [colors[idx] for idx in perm]
        return colors

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):

        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        base_di = self.di(features['res4'])
        # domain specific
        base_ds = self.ds(features['res4'])
        proto = self.pro2(features['res4'])

        basef = torch.cat((base_di, base_ds), dim=1)
        features['res4'] = self.conv_out(basef) + features['res4']
        features['res4'] = self.pro(features['res4'])

        if detected_instances is None:
            if self.proposal_generator is not None:
                logits, proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]


            try:
                results, _ = self.roi_heads(images, features, proposals, None, None, self.backbone)
            except:
                results, _ = self.roi_heads(images, features, proposals, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

            allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

            return allresults
        else:
            return results


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std



@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneWithOffsetGenTrainable(ClipRCNNWithClipBackbone):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # domain_text = {'day': 'an image taken during the day'}
        # # print("f1_dis_p6")
        # with open('prunedprompts.txt', 'r') as f:
        #     for ind, l in enumerate(f):
        #         domain_text.update({str(ind): l.strip()})
        # # self.offsets = nn.Parameter(offsets)
        # #         self.offsets = nn.Parameter(torch.zeros(len(domain_text)-1,1024,14,14)) #skip day

        self.cot_texts = []  # 存储结构: [{'l1': '...', 'l2': '...', 'l3': '...'}, ...]

        try:
            with open('cot1.txt', 'r') as f1, \
                    open('cot2.txt', 'r') as f2, \
                    open('cot3.txt', 'r') as f3:

                lines1 = f1.readlines()
                lines2 = f2.readlines()
                lines3 = f3.readlines()

                assert len(lines1) == len(lines2) == len(lines3), "三个思维链文件的行数必须一致！"

                for l1, l2, l3 in zip(lines1, lines2, lines3):
                    self.cot_texts.append({
                        'l1': l1.strip(),  # 关键词 W
                        'l2': l2.strip(),  # 短语 P
                        'l3': l3.strip()  # 句子 S
                    })
        except FileNotFoundError:
            print("错误: 未找到 cot_level1/2/3.txt 文件，请检查路径。")
            # 这是一个 fallback，防止报错，实际使用请确保文件存在
            self.cot_texts = [{'l1': 'style', 'l2': 'style phrase', 'l3': 'style sentence'}] * 5

        # 参数数量取决于风格的数量 (文件行数)
        num_styles = len(self.cot_texts)

        # self.pin = PIN()
        self.stylemean = nn.Parameter(torch.zeros(num_styles, 1024, 1, 1))
        self.stylestd = nn.Parameter(torch.ones(num_styles, 1024, 1, 1))

        import clip
        #         ipdb.set_trace()
        # self.domain_tk = dict([(k, clip.tokenize(t)) for k, t in domain_text.items()])

        self.cot_tokens = []
        for item in self.cot_texts:
            self.cot_tokens.append({
                'l1': clip.tokenize(item['l1']),
                'l2': clip.tokenize(item['l2']),
                'l3': clip.tokenize(item['l3'])
            })

        self.apply_aug = cfg.AUG_PROB
        self.pro = Singe_prototype(1024,7)
        self.pro2 = Singe_prototype(1024, 7)

        self.final_prototype_module = PrototypeClusteringModule(
            in_channels=1024,
            num_classes=7  # 根据您的任务类别数调整
        )

        self.di = invariant()
        self.ds = specific()
        self.conv_out = convblock(2 * 1024, 1024, 3, 1, 1)
        self.consistency_loss = ConsistencyLoss()

    def forward(self, batched_inputs):
        #         idpb.set_trace()

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        base_di = self.di(features['res4'])
        # domain specific
        base_ds = self.ds(features['res4'])
        proto =  self.final_prototype_module(features['res4'])
        # Sequential Once
        lossp = self.consistency_loss(proto, base_di)
        zero_loss_di_p = F.normalize(base_di) * F.normalize(features['res4'])
        zero_loss_di_n = F.normalize(base_ds) * F.normalize(features['res4'])

        zero_loss_di_p = torch.exp(torch.sum(zero_loss_di_p, dim=1))
        zero_loss_di_n = torch.exp(torch.sum(zero_loss_di_n, dim=1))

        log_result_di = torch.log(zero_loss_di_p / (zero_loss_di_p + zero_loss_di_n)) * -1.0
        zero_loss_di = torch.mean(log_result_di)

        if np.random.rand(1) > self.apply_aug:
            oids = np.random.choice(np.arange(len(self.stylemean)), b)
            mean = torch.cat([self.stylemean[oid:oid + 1].cuda().mean(dim=(2, 3), keepdims=True) for oid in oids], 0)
            std = torch.cat([self.stylestd[oid:oid + 1].cuda().mean(dim=(2, 3), keepdims=True) for oid in oids], 0)
            base_ds = base_ds * std.expand(base_ds.size()) + mean.expand(
                base_ds.size())
        basef = torch.cat((base_di, base_ds), dim=1)
        features['res4'] = self.conv_out(basef) + features['res4']
        # features['res4'] = self.pro(features['res4'])
        features['res4'] = self.final_prototype_module(features['res4'])

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        loss_di = {'zero_loss_di': zero_loss_di}
        loss_dp = {'loss_dp': lossp}
        losses = {}
        losses.update(loss_dp)
        losses.update(loss_di)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def opt_offsets(self, batched_inputs):

        crops_clip = None
        if 'randomcrops' in batched_inputs[0]:
            rcrops = [x['randomcrops'] for x in batched_inputs]
            rcrops = torch.cat(rcrops, 0)
            crops_clip = rcrops.flip(1) / 255
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            crops_clip = T.functional.normalize(crops_clip, mean, std)
            crops_clip = crops_clip.cuda()

        with torch.no_grad():
            features = self.backbone(crops_clip)

        losses = {}
        total_dist = 0
        total_reg = 0
        # total_chgn = 0
        clip_model = self.roi_heads.box_predictor.cls_score.model  # 获取 CLIP 模型实例

        for i, tokens in enumerate(self.cot_tokens):
            # --- 1. 计算思维链文本特征 Ft3 ---
            with torch.no_grad():
                # Eq (1): Ft1 = E(W)
                ft1 = clip_model.encode_text(tokens['l1'].cuda())
                # print("生成f1")

                # Eq (2): Ft2 = E(P) + Ft1
                ft2 = clip_model.encode_text(tokens['l2'].cuda()) + ft1
                # print("生成f2")
                # Eq (3): Ft3 = E(S) + Ft2
                ft3 = clip_model.encode_text(tokens['l3'].cuda()) + ft2
                # print("生成f3")
                # 归一化 Ft3，作为最终的文本指导信号
                ft3_norm = ft3 / ft3.norm(dim=-1, keepdim=True)

                # 调整维度以匹配 calculate consistency (N, C) -> (C, 1) or similar depending on implementation
                # CLIP 输出通常是 (1, 1024)，这里我们需要它作为 target
                target_text_feat = ft3_norm  # (1, 1024)

            # --- 2. 视觉特征风格演化 (Style Evolution) ---
            # Eq (5): Fi = sigma_t(Fs') + mu_t
            # features['res4'] 对应 Fs

            # 获取对应的 learnable parameters
            current_std = self.stylestd[i].expand(features['res4'].size())
            current_mean = self.stylemean[i].expand(features['res4'].size())

            # 应用 AdaIN 风格变换
            # Fs' = (Fs - mu) / sigma 这一步在 AdaIN 通常隐含在 Norm 层中，
            # 但在这里代码直接对 raw features 进行 scale 和 shift。
            # 假设 features['res4'] 已经是比较原始的状态，我们直接应用变换生成 Fi
            aug_feat = features['res4'] * current_std + current_mean  # Fi

            # --- 3. 提取演化后的视觉特征向量 ---
            # 为了计算和文本的相似度，我们需要通过 Backbone 后半部分得到全局特征
            x = self.backbone.forward_res5(aug_feat)
            im_embed = self.backbone.attention_global_pool(x)  # (Batch, 1024)
            im_embed = im_embed / im_embed.norm(dim=-1, keepdim=True)  # 归一化

            # --- 4. 计算一致性损失 Ltc ---
            # Eq (6): Ltc = 1 - sim(Fi, Ft3)
            # 计算 cosine similarity
            # target_text_feat: (1, 1024), im_embed: (Batch, 1024)

            # 矩阵乘法计算相似度: (Batch, 1024) * (1024, 1) -> (Batch, 1)
            similarity = torch.mm(im_embed, target_text_feat.transpose(0, 1))

            # Loss: 1 - mean(similarity)
            loss_tc = 1.0 - similarity.mean()

            total_dist += loss_tc

        # 更新 Loss dict
        # 除以 len(self.cot_tokens) 得到平均 Loss
        losses.update({
            'loss_style_consistency': total_dist / len(self.cot_tokens)
        })

        # ---------------------- 修改结束 ----------------------

        return losses







        # for i, val in enumerate(self.domain_tk.items()):
        #     name, dtk = val
        #     # print(name)
        #     # print(dtk)
        #     if name == 'day':
        #          continue
        #     with torch.no_grad():
        #         # print(self.backbone.forward_res5(features['res4']))
        #         # print(name)
        #         wo_aug_im_embed = self.backbone.attention_global_pool(self.backbone.forward_res5(features['res4']))
        #         wo_aug_im_embed = wo_aug_im_embed / wo_aug_im_embed.norm(dim=-1, keepdim=True)
        #
        #         day_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(
        #             self.domain_tk['day'].cuda())  # day
        #         day_text_embed = day_text_embed / day_text_embed.norm(dim=-1, keepdim=True)
        #         new_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(dtk.cuda())  # new_d
        #         new_text_embed = new_text_embed / new_text_embed.norm(dim=-1, keepdim=True)
        #         text_off = (new_text_embed - day_text_embed)
        #         text_off = text_off / text_off.norm(dim=-1, keepdim=True)
        #
        #         wo_aug_im_tsl = wo_aug_im_embed + text_off
        #         wo_aug_im_tsl = wo_aug_im_tsl / wo_aug_im_tsl.norm(dim=-1, keepdim=True)
        #         wo_aug_im_tsl = wo_aug_im_tsl.unsqueeze(1).permute(0, 2, 1)
        #
        #     # aug_feat = features['res4'].detach() + self.offsets[i - 1:i]
        #     aug_feat = features['res4'] * self.stylestd[i - 1].expand(features['res4'].size()) + self.stylemean[
        #       i - 1].expand(features['res4'].size())
        #
        #     x = self.backbone.forward_res5(aug_feat)
        #     im_embed = self.backbone.attention_global_pool(x)
        #
        #     im_embed = im_embed / im_embed.norm(dim=-1, keepdim=True)
        #
        #     cos_dist = 1 - im_embed.unsqueeze(1).bmm(wo_aug_im_tsl)
        #
        #     dist_loss = cos_dist.mean()
        #
        #     l1loss = torch.nn.functional.l1_loss(im_embed, wo_aug_im_embed)
        #
        #     total_dist += dist_loss
        #     total_reg += l1loss
        #
        # losses.update({f'cos_dist_loss_{name}': total_dist / len(self.domain_tk),
        #                f'reg_loss_{name}': total_reg / len(self.domain_tk)})
        #
        # return losses


