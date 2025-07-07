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


class ClassSpecificPrototypeClustering(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.conv_prototype = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features):
        """

        """
        B, C, H, W = features.shape
        assert C == self.feature_dim, "特征维度不匹配"


        prob = F.softmax(self.conv_prototype(features), dim=1)  # [B, K, H, W]


        features_flat = features.view(B, C, -1)  # [B, C, H*W]
        prob_flat = prob.view(B, self.num_classes, -1)  # [B, K, H*W]


        prototypes = torch.bmm(prob_flat, features_flat.transpose(1, 2))  # [B, K, C]
        prototypes = F.normalize(prototypes, dim=2)  # 归一化

    
        self.current_prototypes = prototypes
        return features

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

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
        self.pro = Singe_prototype(1024, 8)
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
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        base_di = self.di(features['res4'])
        # domain specific
        base_ds = self.ds(features['res4'])


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


class ChainOfThoughtPromptGenerator:
    def __init__(self, base_vocabs, clip_model):
        """

        """
        self.vocabs = base_vocabs
        self.clip_model = clip_model
        self.stage1_vocab = {}
        self.stage2_phrases = {}
        self.stage3_sentences = {}
        self._build_prompts()

    def _build_prompts(self):

        for key, words in self.vocabs.items():
            self.stage1_vocab[key] = words


        weather = self.vocabs.get('weather', ['sunny'])
        time = self.vocabs.get('time', ['day'])
        action = self.vocabs.get('action', ['driving'])
        for w in weather:
            for t in time:
                for a in action:
                    phrase = f"{w} {t} {a}"
                    self.stage2_phrases[phrase] = f"A style of '{phrase}'"

        details = self.vocabs.get('detail', ['vehicles on the road', 'pedestrians'])
        for phrase in self.stage2_phrases:
            for d in details:
                sentence = f"{phrase} with {d}"
                self.stage3_sentences[sentence] = f"A detailed description of '{sentence}'"

    def generate_evolution_features(self, stage=3):

        all_text_features = []


        if stage >= 1:
            stage1_texts = []
            for key, words in self.stage1_vocab.items():
                stage1_texts.extend([f"A {key} style: '{w}'" for w in words])
            stage1_tokens = clip.tokenize(stage1_texts).to(self.clip_model.device)
            with torch.no_grad():
                stage1_features = self.clip_model.encode_text(stage1_tokens)
                stage1_features = stage1_features.mean(dim=0, keepdim=True)  # 合并基础特征
            all_text_features.append(stage1_features)


        if stage >= 2:
            stage2_tokens = clip.tokenize(list(self.stage2_phrases.values())).to(self.clip_model.device)
            with torch.no_grad():
                stage2_features = self.clip_model.encode_text(stage2_tokens)
                stage2_features = stage2_features.mean(dim=0, keepdim=True)
            all_text_features.append(stage2_features)

        if stage >= 3:
            stage3_tokens = clip.tokenize(list(self.stage3_sentences.values())).to(self.clip_model.device)
            with torch.no_grad():
                stage3_features = self.clip_model.encode_text(stage3_tokens)
                stage3_features = stage3_features.mean(dim=0, keepdim=True)
            all_text_features.append(stage3_features)


        evolved_features = all_text_features[0]
        for feat in all_text_features[1:]:
            evolved_features += feat
        return evolved_features

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneWithOffsetGenTrainable(ClipRCNNWithClipBackbone):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)


        base_vocabs = {
            'weather': ['rainy', 'foggy', 'sunny'],
            'time': ['night', 'dusk', 'day'],
            'action': ['driving', 'parking'],
            'detail': ['vehicles on road', 'pedestrians', 'traffic lights']
        }
        self.cot_prompt_generator = ChainOfThoughtPromptGenerator(base_vocabs, self.backbone.clip_model)


        self.style_evolution_stages = cfg.STYLE_EVOLUTION_STAGES or 3  # 演化阶段数
        self.style_params = nn.ModuleList([
            nn.Parameter(torch.zeros(1, 1024, 1, 1)),  # 均值
            nn.Parameter(torch.ones(1, 1024, 1, 1))  # 标准差
        ])

        self.di = invariant()
        self.ds = specific()
        # self.style_extractor = nn.Conv2d(1024, 1024, kernel_size=1)
        # self.content_extractor = nn.Conv2d(1024, 1024, kernel_size=1)
        self.contrastive_loss = nn.CrossEntropyLoss()


        self.prototype_clustering = ClassSpecificPrototypeClustering(1024, num_classes=8)  # 假设8个类别

    def forward(self, batched_inputs):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        base_features = features['res4']


        style_features = self.ds(base_features)
        content_features = self.di(base_features)


        evolved_text_features = self.cot_prompt_generator.generate_evolution_features(
            stage=self.style_evolution_stages
        )
        evolved_text_features = evolved_text_features.view(1, -1, 1, 1).expand(
            style_features.size(0), -1, style_features.size(2), style_features.size(3)
        )


        mean, std = self.style_params

        mean = mean + 0.1 * evolved_text_features.mean(dim=(2, 3), keepdims=True)
        std = std + 0.1 * evolved_text_features.std(dim=(2, 3), keepdims=True)


        normalized_style = (style_features - style_features.mean(dim=(2, 3), keepdims=True)) / (
                style_features.std(dim=(2, 3), keepdims=True) + 1e-5
        )
        styled_features = std * normalized_style + mean


        fused_features = torch.cat([content_features, styled_features], dim=1)
        fused_features = self.conv_out(fused_features) + base_features
        features['res4'] = self.pro(fused_features)


        return super().forward(batched_inputs)

    def opt_offsets(self, batched_inputs):

        for stage in range(1, self.style_evolution_stages + 1):

            stage_features = self.cot_prompt_generator.generate_evolution_features(stage=stage)
            stage_features = stage_features.view(1, -1, 1, 1).cuda()


            mean, std = self.style_params
            style_repr = std.mean(dim=(2, 3)) + mean.mean(dim=(2, 3))
            text_repr = stage_features.mean(dim=(2, 3))
            consistency_loss = 1 - F.cosine_similarity(style_repr, text_repr, dim=1)

            losses[f'cot_consistency_stage{stage}'] = consistency_loss.mean()

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
        total_chgn = 0
        for i, val in enumerate(self.domain_tk.items()):
            name, dtk = val
            # print(name)
            # print(dtk)
            if name == 'day':
                 continue
            with torch.no_grad():
                # print(self.backbone.forward_res5(features['res4']))
                # print(name)
                wo_aug_im_embed = self.backbone.attention_global_pool(self.backbone.forward_res5(features['res4']))
                wo_aug_im_embed = wo_aug_im_embed / wo_aug_im_embed.norm(dim=-1, keepdim=True)

                day_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(
                    self.domain_tk['day'].cuda())  # day
                day_text_embed = day_text_embed / day_text_embed.norm(dim=-1, keepdim=True)
                new_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(dtk.cuda())  # new_d
                new_text_embed = new_text_embed / new_text_embed.norm(dim=-1, keepdim=True)
                text_off = (new_text_embed - day_text_embed)
                text_off = text_off / text_off.norm(dim=-1, keepdim=True)

                wo_aug_im_tsl = wo_aug_im_embed + text_off
                wo_aug_im_tsl = wo_aug_im_tsl / wo_aug_im_tsl.norm(dim=-1, keepdim=True)
                wo_aug_im_tsl = wo_aug_im_tsl.unsqueeze(1).permute(0, 2, 1)

            # aug_feat = features['res4'].detach() + self.offsets[i - 1:i]
            aug_feat = features['res4'] * self.stylestd[i - 1].expand(features['res4'].size()) + self.stylemean[
              i - 1].expand(features['res4'].size())

            x = self.backbone.forward_res5(aug_feat)
            im_embed = self.backbone.attention_global_pool(x)

            im_embed = im_embed / im_embed.norm(dim=-1, keepdim=True)

            cos_dist = 1 - im_embed.unsqueeze(1).bmm(wo_aug_im_tsl)

            dist_loss = cos_dist.mean()

            l1loss = torch.nn.functional.l1_loss(im_embed, wo_aug_im_embed)

            total_dist += dist_loss
            total_reg += l1loss

        losses.update({f'cos_dist_loss_{name}': total_dist / len(self.domain_tk),
                       f'reg_loss_{name}': total_reg / len(self.domain_tk)})

        return losses




