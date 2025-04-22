from transformers import RobertaForSequenceClassification
from peft import get_peft_model, LoraConfig

def create_model(config, num_labels, id2label):
    # Initialize base model
    model = RobertaForSequenceClassification.from_pretrained(
        config['model']['base_model'],
        num_labels=num_labels,
        id2label=id2label
    )
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        target_modules=config['lora']['target_modules'],
        task_type="SEQ_CLS"
    )
    
    # Get PEFT model
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model
