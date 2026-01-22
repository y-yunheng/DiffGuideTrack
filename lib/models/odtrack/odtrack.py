import torch
from torch import nn
from torch.nn import functional as F
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
# from lib.models.odtrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.odtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
# =============================================================================
# [新改进] 尺度自适应注意力模块 (Scale-Adaptive Attention Module)
# 理论改进点：
# 1. 双流分支 (Dual-Stream): 分别捕捉微小目标(3x3)和快速/大尺度目标(Dilation=3)
# 2. 空间注意力 (Spatial Attention): 替代 Concat，通过 Sigmoid 自动抑制背景噪声
# 3. 瓶颈结构 (Bottleneck): 减少参数量，防止过拟合
# =============================================================================
class InvertedDifferenceFusion(nn.Module):
    def __init__(self, dim, feat_sz, expand_ratio=2.0):
        super().__init__()
        self.feat_sz = feat_sz
        hidden_dim = int(dim * expand_ratio)
        
        # 1. 升维 (Point-wise Conv): 增加特征通道，给小目标更多的"存活空间"
        self.expand_conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # 2. 并行差分提取 (Parallel Extraction)
        # 分支 A: 3x3 深度卷积 (捕捉 3x3 极小点)
        self.dw_conv_3x3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        
        # 分支 B: 5x5 深度卷积 (捕捉稍大目标或背景上下文)
        # 5x5 对于 CST 数据集的近距离目标更友好，且能提供背景参考
        self.dw_conv_5x5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        
        # 3. 差分增强 (Difference Enhancement)
        # 这是一个简单的注意力：我们让网络学习 3x3 和 5x5 的加权组合
        # 理论上，如果是小目标，3x3响应强，5x5响应弱，二者差异大。
        self.fusion_weight = nn.Parameter(torch.ones(1, hidden_dim, 1, 1) * 0.5)

        self.act = nn.GELU()
        
        # 4. 降维投影 (Projection): 融合信息并还原通道数
        self.proj_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 5. 学习一个缩放系数，初始化为很小，保证初始训练稳定
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x_vit):
        # x_vit: [B, HW, C]
        B, HW, C = x_vit.shape
        H = W = self.feat_sz
        
        # 变换为图像 [B, C, H, W]
        x_img = x_vit.transpose(1, 2).reshape(B, C, H, W)
        
        # 1. 升维
        x_high = self.expand_conv(x_img)
        
        # 2. 多尺度特征
        x_3x3 = self.dw_conv_3x3(x_high)
        x_5x5 = self.dw_conv_5x5(x_high)
        
        # 3. 软差分融合
        # 类似于 Center-Surround 机制。
        # 如果 fusion_weight 变为负值，网络实际上就在做 x_3x3 - x_5x5 (高通滤波)
        print("self.fusion_weight:  ",self.fusion_weight)
        x_fused = x_3x3 + self.fusion_weight * (x_5x5 - x_3x3)
        x_fused = self.act(x_fused)
        
        # 4. 投影回原始维度
        x_out = self.proj_conv(x_fused)
        
        # 5. 残差连接 (带可学习的缩放 gamma)
        # [B, C, H, W] -> [B, HW, C]
        x_out = x_out.flatten(2).transpose(1, 2)
        
        return x_vit + self.gamma * x_out,x_high, x_3x3, x_5x5


# 新增：动态特征路由模块 - 放在InvertedDifferenceFusion模块之后
class DynamicFeatureRouter(nn.Module):#TGG
    def __init__(self, dim, feat_sz):
        super().__init__()
        self.feat_sz = feat_sz
        # 针对模板全局信息的处理
        self.query_gate = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.GELU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid()
        )

    def forward(self, search_vit, search_refined, target_global):
        """
        target_global: 来自模板的全局精髓 [B, 1, C]
        """
        B, HW, C = search_vit.shape
        # 1. 核心改进：利用模板信息生成“全局目标过滤器”
        # 这是一个内容相关的 Gate，它告诉网络：哪些通道是这个目标特有的
        gate = self.query_gate(target_global) # [B, 1, C]
        
        # 2. 用模板的先验去加权搜索区域的特征
        # 这才叫真正的“全局信息输入”，因为它引入了搜索区域之外的、确定性的目标信息
        filtered_feat = search_refined * gate 
        
        return search_vit + filtered_feat,gate

class ODTrack(nn.Module):
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.track_query = None
        self.token_len = token_len

        # =================================================================
        # [改进点一] 使用 v2 版反向差分融合模块
        hidden_dim = transformer.embed_dim 
        # expand_ratio=2.0 意味着中间层通道翻倍，给小目标更多特征空间
        self.fusion_module = InvertedDifferenceFusion(hidden_dim, self.feat_sz_s, expand_ratio=2.0)
        # =================================================================
        # [改进点二] 动态特征路由模块 - 不影响现有改进，但增加全局感知
        self.dynamic_router = DynamicFeatureRouter(hidden_dim, self.feat_sz_s)

    def forward(self, template: torch.Tensor, search: torch.Tensor,
                ce_template_mask=None, ce_keep_rate=None, return_last_attn=False):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):
            # 1. Backbone 输出 x 形状: [B, 1(query) + lens_z(模板) + lens_x(搜索), C]
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
            
            # --- 关键修正 A：精确切分特征流 ---
            # 提取搜索区域 token (最后 feat_len_s 个)
            enc_opt = x[:, -self.feat_len_s:] 
            
            # 提取模板区域 token (在 query 和 搜索区域 之间)
            # lens_z = 总长度 - 1(query) - 搜索区域长度
            lens_z = x.shape[1] - self.token_len - self.feat_len_s
            template_tokens = x[:, self.token_len : self.token_len + lens_z]
            
            # 算出模板全局精髓 (Global Target Descriptor) -> [B, 1, C]
            target_global = template_tokens.mean(dim=1, keepdim=True)
            # ----------------------------------

            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach()

            # --- 改进点一：局部差分增强 ---
            enc_opt_refined,x_high, x_3x3, x_5x5 = self.fusion_module(enc_opt)
            
            # --- 关键修正 B：模板引导的动态路由 ---
            # 必须传入 target_global，否则这个模块只是在“闭门造车”
            enc_opt_fused,gate= self.dynamic_router(enc_opt, enc_opt_refined, target_global)
            # ----------------------------------

            # 计算注意力矩阵用于融合 (保持原版逻辑)
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2)) 
            
            # --- 关键修正 C：确保最终输出使用了 Fused 特征 ---
            # 这里必须使用 enc_opt_fused，而不是之前的中间变量
            opt = (enc_opt_fused.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
            
            out = self.forward_head(opt, None)
            out.update(aux_dict)
            
            out['backbone_feat'] = x
            out['visualization_data']={
                'x_high': x_high.detach().cpu().numpy(),
                'x_3x3': x_3x3.detach().cpu().numpy(),
                'x_5x5': x_5x5.detach().cpu().numpy(),
                'x_diff': (x_5x5 - x_3x3).detach().cpu().numpy(),
                'gate': gate.detach().cpu().numpy(),
                # 原始
                'feat_1_input': enc_opt.detach().cpu().numpy(),
                # 阶段2: IDF 增强后
                'feat_2_idf': enc_opt_refined.detach().cpu().numpy(),
                
                # 阶段3: DFR 过滤后 (最终输出)
                'feat_3_dfr': enc_opt_fused.detach().cpu().numpy(),
            }
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        # 保持不变
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map}
            return out

        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

# Build 函数保持不变
def build_odtrack(cfg, training=True):
    # ... (保持原样，省略以节省空间) ...
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )
    else:
        raise NotImplementedError
        
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ODTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    return model