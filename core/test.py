import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from .preprocess import AGNewsDataset
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from .model import create_model
def test_model(config, experiment_dir, id2label):
    # Load model and tokenizer 
    # Load test data
    test_data = pd.read_pickle('data/test_unlabelled.pkl')
    # Preprocess test data
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['base_model'])
    model =  RobertaForSequenceClassification.from_pretrained(config['model']['base_model'],id2label=id2label)
    test_dataset = AGNewsDataset(tokenizer(
        test_data['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=config['data']['max_length']
    ))
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['testing']['batch_size'],
        shuffle=False
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Run inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().numpy())
    
    # Save predictions
    df_output = pd.DataFrame({
        'ID': range(len(predictions)),
        'Label': predictions
    })
    df_output.to_csv(os.path.join(experiment_dir, 'predictions.csv'), index=False)
    
    # Log completion
    with open(os.path.join(experiment_dir, 'execution_log.txt'), 'a') as f:
        f.write("\nTesting completed. Predictions saved to predictions.csv\n")
