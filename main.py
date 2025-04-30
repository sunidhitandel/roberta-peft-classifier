import os
import argparse
import yaml
from core.trainer import Trainer
from core.test import test_model

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate RoBERTa with LoRA')
    parser.add_argument('--config', type=str, required=True, help='Name of the config file (without .yaml)')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join('configs', f'{args.config}.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create experiment directory
    experiment_dir = os.path.join('experiments', args.config)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize trainer and train
    trainer, id2label  = Trainer(config, experiment_dir)
    trainer.train()

    # Test the model
    test_model(config, experiment_dir, id2label)

if __name__ == '__main__':
    main()
