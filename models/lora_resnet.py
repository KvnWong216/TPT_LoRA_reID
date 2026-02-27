import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model

class TPTLoRAResNet(nn.Module):
    def __init__(self, num_classes=2, r=2, alpha=1): 
        super().__init__()
        base_model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["layer4"], # Higher layer for semantic feature extraction. 
            lora_dropout=0.1,
            bias="none"
        )
    
        self.lora_model = get_peft_model(base_model, config)
        self.core_model = self.lora_model.base_model.model

    def forward(self, x, return_embeddings=False):
        features = self.core_model.forward_features(x)
        embeddings = self.core_model.forward_head(features, pre_logits=True)
        
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        logits = self.core_model.fc(embeddings)
        
        if return_embeddings:
            return logits, normalized_embeddings
        return logits

    def print_trainable_parameters(self):
        self.lora_model.print_trainable_parameters()
        
    def save_pretrained(self, save_directory):
        self.lora_model.save_pretrained(save_directory)

def get_lora_res(num_classes=2, r=2, alpha=1):
    return TPTLoRAResNet(num_classes=num_classes, r=r, alpha=alpha)