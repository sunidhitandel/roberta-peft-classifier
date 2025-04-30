from datasets import load_dataset
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset

class AGNewsDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.data.items()}

def preprocess_data(config):
    # Load AG News dataset
    dataset = load_dataset('ag_news', split='train')
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['base_model'])
    
    # Preprocess function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=config['data']['max_length']
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Rename label column
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    
    # Get number of classes and class names
    num_labels = dataset.features['label'].num_classes
    class_names = dataset.features['label'].names
    
    # Create id2label mapping
    id2label = {i: label for i, label in enumerate(class_names)}
    
    # Split dataset
    split_datasets = tokenized_dataset.train_test_split(
        test_size=config['data']['test_size'],
        seed=config['data']['seed']
    )
    
    return {
        'train_dataset': AGNewsDataset(split_datasets['train']),
        'eval_dataset': AGNewsDataset(split_datasets['test']),
        'tokenizer': tokenizer,
        'num_labels': num_labels,
        'id2label': id2label
    }
