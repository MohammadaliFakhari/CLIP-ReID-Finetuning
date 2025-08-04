import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MaPLeTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text, cls_ctx_per_id):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, compound_prompts_deeper_text, cls_ctx_per_id, 0]
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.trainer = 'MaPLe'
        self.use_lora = cfg.MODEL.USE_LORA
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size, self.trainer)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        if self.use_lora:
            self.list_lora_layers = apply_lora(cfg, self.image_encoder, clip_model)
        # if cfg.eval_only:
        #     load_lora(cfg, list_lora_layers)

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        if self.trainer == 'MaPLe':
            self.prompt_learner = MaplePromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
            self.text_encoder = MaPLeTextEncoder(clip_model)
        else:
            self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
            self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, pids=None, all_gender=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        maple = True if self.trainer == 'MaPLe' else False
        if get_text == True:
            if maple:
                prompts, deep_compound_prompts_text, cls_ctx_per_id = self.prompt_learner(label)
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts, deep_compound_prompts_text, cls_ctx_per_id)
                return text_features
            else:
                prompts = self.prompt_learner(label) 
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
                return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_emb=cv_embed)
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size, trainer):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    design_details = {"trainer": trainer,
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": 2,
                      "maple_length_per_id": 1}

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size, design_details)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 

class MaplePromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        self.is_TeD = True
        if self.is_TeD:
            ctx_init = "A photo of a Y Y X person."
            ctx_dim = 512
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = 4
            n_cls_ctx_per_id = 1
            n_cls_ctx = 2
        else:
            ctx_init = "A photo of a Y Y X X X person."
            ctx_dim = 50
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = 4
            n_cls_ctx_per_id = 3
            n_cls_ctx = 2
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        # -------------------------------------------------------------------------------------
        cls_vectors_per_id = torch.empty(num_class, n_cls_ctx_per_id, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors_per_id, std=0.02)
        self.cls_ctx_per_id = nn.Parameter(cls_vectors_per_id)
        # -------------------------------------------------------------------------------------

        cls_vector = torch.empty(n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vector, std=0.02)
        self.cls_vector = nn.Parameter(cls_vector)

        print('TeD design: Text-Deep Prompt Learning')

        self.compound_prompts_depth = 9
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_cls_ctx, ctx_dim, dtype=dtype))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for prompt in self.compound_prompts_text:
            nn.init.normal_(prompt, std=0.02)

        # -------------------------------------------------------------------------------------
        self.compound_per_id_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(num_class, n_cls_ctx_per_id, ctx_dim, dtype=dtype))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for prompt in self.compound_per_id_prompts_text:
            nn.init.normal_(prompt, std=0.02)
        # -------------------------------------------------------------------------------------
        if not self.is_TeD:
            self.proj_per_id = nn.Linear(ctx_dim, 512)
            self.text_proj = nn.Linear(ctx_dim, 512)
            self.compound_prompt_projections = nn.ModuleList([
                nn.Linear(ctx_dim, 512)
                for _ in range(self.compound_prompts_depth - 1)
            ])
            self.compound_prompt_projections_per_id = nn.ModuleList([
                nn.Linear(ctx_dim, 512)
                for _ in range(self.compound_prompts_depth - 1)
            ])
        # -------------------------------------------------------------------------------------


        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx + n_cls_ctx_per_id: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx_per_id = self.cls_ctx_per_id[label]
        b = label.shape[0]
        if self.cls_vector.dim() == 2:
            cls_vector = self.cls_vector.unsqueeze(0).expand(b, -1, -1)
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 

        if not self.is_TeD:
            cls_vector = self.text_proj(cls_vector)
            cls_ctx_per_id = self.proj_per_id(cls_ctx_per_id)
        
        prompts = torch.cat(
            [
                prefix,
                cls_vector,
                cls_ctx_per_id,
                suffix,
            ],
            dim=1,
        )
        
        text_deep_per_id_prompts = []
        compound_prompts_text = []
        if not self.is_TeD:
            for i, layer in enumerate(self.compound_prompt_projections):
                compound_prompts_text.append(layer(self.compound_prompts_text[i]))
            for i, layer in enumerate(self.compound_prompt_projections_per_id):
                text_deep_per_id_prompts.append(layer(self.compound_per_id_prompts_text[i][label]))
        else:
            for i in range(self.compound_prompts_depth-1):
                text_deep_per_id_prompts.append(self.compound_per_id_prompts_text[i][label])
            for i in range(self.compound_prompts_depth-1):
                compound_prompts_text.append((self.compound_prompts_text[i]))
        # --------------------------------------------------------------------------------
        return prompts, compound_prompts_text, text_deep_per_id_prompts
