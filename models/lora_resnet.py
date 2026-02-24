import timm
from peft import LoraConfig, get_peft_model

def get_lora_res(num_classes=751, r=8, alpha=16):
    model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["conv2"],
        lora_dropout=0.1,
        bias="none"
    )

    lora_model = get_peft_model(model, config)
    return lora_model