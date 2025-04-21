import logging
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)

class TrainerHandler:
    def __init__(self, config, tokenized_dataset, model):
        self.config = config
        self.tokenized_dataset = tokenized_dataset
        self.model = model

    def train(self):
        logger.info("Preparing training arguments...")
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            evaluation_strategy="steps",
            logging_dir=self.config['output_dir'],
            logging_steps=100,
            save_steps=500,
            learning_rate=self.config['learning_rate'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['train_batch_size'],
            per_device_eval_batch_size=self.config['eval_batch_size'],
        )

        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed.")