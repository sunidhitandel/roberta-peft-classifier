import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import numpy as np
from .preprocess import preprocess_data
from .model import create_model

class Trainer:
    def __init__(self, config, experiment_dir):
        self.config = config
        self.experiment_dir = experiment_dir
        self.setup_logging()
        
        # Preprocess data
        data = preprocess_data(config)
        self.train_dataset = data['train_dataset']
        self.eval_dataset = data['eval_dataset']
        self.tokenizer = data['tokenizer']
        
        # Create model
        self.model = create_model(config, data['num_labels'], data['id2label'])
        self.id2label = data['id2label']
        # Setup training arguments
        self.training_args = TrainingArguments(
            output_dir=experiment_dir,
            report_to=None,
            use_cpu=False,
            evaluation_strategy='steps',
            logging_steps=config['training']['logging_steps'],
            learning_rate=config['training']['learning_rate'],
            num_train_epochs=config['training']['num_epochs'],
            max_steps=config['training']['max_steps'],
            per_device_train_batch_size=config['training']['train_batch_size'],
            per_device_eval_batch_size=config['training']['eval_batch_size'],
            optim=config['training']['optimizer'] if config['training']['optimizer'] else None,
            gradient_checkpointing=config['training']['gradient_checkpointing'],
            gradient_checkpointing_kwargs={'use_reentrant': True},
            weight_decay=config['training']['weight_decay'],
            dataloader_num_workers=config['training']['dataloader_num_workers'] if config['training']['dataloader_num_workers'] else None,
            lr_scheduler_type=config['training']['lr_scheduler_type'] if config['training']['lr_scheduler_type'] else None,
            warmup_ratio=config['training']['warmup_ratio'] if config['training']['warmup_ratio'] else None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            warmup_steps=config['training']['warmup_steps'] if config['training']['warmup_steps'] else None,
            greater_is_better=False
        )
    
        # Setup data collator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
    
    def setup_logging(self):
        self.log_file = os.path.join(self.experiment_dir, 'execution_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Starting training with config:\n{self.config}\n\n")
    
    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        return {'accuracy': accuracy}
    
    def train(self):
        self.log("Starting training...")
        train_result = self.trainer.train()
        
        # Save model and tokenizer
        self.model.save_pretrained(os.path.join(self.experiment_dir, 'model'))
        self.tokenizer.save_pretrained(os.path.join(self.experiment_dir, 'tokenizer'))
        
        # Log training results
        self.log(f"Training completed. Results: {train_result}")
        
        return train_result,self.id2label 