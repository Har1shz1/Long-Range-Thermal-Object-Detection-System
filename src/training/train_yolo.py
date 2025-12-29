"""
YOLOv5 Training Script for Thermal Object Detection
"""

import torch
import yaml
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Optional

# Add YOLOv5 to path
yolov5_path = Path(__file__).parent.parent.parent / 'yolov5'
if yolov5_path.exists():
    sys.path.append(str(yolov5_path))
else:
    print("Downloading YOLOv5...")
    os.system('git clone https://github.com/ultralytics/yolov5.git')
    sys.path.append('yolov5')

from models.common import DetectMultiBackend
from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import (
    check_img_size, check_file, increment_path, 
    colorstr, print_args, set_logging
)
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
from utils.loggers import Loggers
from val import run as val_run

class ThermalYOLOTrainer:
    """Trainer class for YOLOv5 on thermal images"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_device()
        self.setup_paths()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set defaults
        defaults = {
            'weights': 'yolov5s.pt',
            'data': 'data/dataset.yaml',
            'epochs': 100,
            'batch_size': 16,
            'imgsz': 640,
            'device': '',
            'workers': 8,
            'project': 'runs/train',
            'name': 'exp',
            'exist_ok': False,
            'patience': 50,
            'save_period': -1,
            'eval_period': 1,
            'entity': None,
            'upload_dataset': False,
            'bbox_interval': -1,
            'artifact_alias': 'latest'
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def setup_logging(self):
        """Setup logging configuration"""
        set_logging()
        self.logger = logging.getLogger(__name__)
        
        log_dir = Path(self.config['project']) / self.config['name']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Training configuration: {self.config}")
    
    def setup_device(self):
        """Setup training device (GPU/CPU)"""
        self.device = select_device(self.config['device'])
        self.logger.info(f"Using device: {self.device}")
        
        # Mixed precision training
        self.amp = True if self.device.type != 'cpu' else False
        
    def setup_paths(self):
        """Setup directory paths"""
        # Increment run directory
        save_dir = increment_path(
            Path(self.config['project']) / self.config['name'],
            exist_ok=self.config['exist_ok']
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        
        # Save config to run directory
        config_save_path = save_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.logger.info(f"Results saved to {save_dir}")
    
    def prepare_data(self):
        """Prepare data loaders"""
        data_dict = self.check_dataset()
        
        # Create train dataloader
        train_loader = create_dataloader(
            data_dict['train'],
            imgsz=self.config['imgsz'],
            batch_size=self.config['batch_size'],
            stride=32,
            single_cls=False,
            pad=0.5,
            rect=False,
            workers=self.config['workers'],
            prefix=colorstr('train: '),
        )[0]
        
        # Create validation dataloader
        val_loader = create_dataloader(
            data_dict['val'],
            imgsz=self.config['imgsz'],
            batch_size=self.config['batch_size'] * 2,
            stride=32,
            single_cls=False,
            pad=0.5,
            rect=True,
            workers=self.config['workers'],
            prefix=colorstr('val: '),
        )[0]
        
        return train_loader, val_loader, data_dict
    
    def check_dataset(self) -> Dict:
        """Check and load dataset configuration"""
        data = self.config['data']
        
        with open(data, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        
        # Verify paths exist
        for split in ['train', 'val', 'test']:
            if split in data_dict:
                path = Path(data_dict[split])
                if not path.exists():
                    raise FileNotFoundError(f"Dataset path not found: {path}")
        
        # Number of classes
        nc = int(data_dict['nc'])
        assert nc > 0, f"Number of classes must be positive, got {nc}"
        
        # Class names
        names = data_dict['names']
        assert len(names) == nc, f"Number of names {len(names)} != nc {nc}"
        
        self.logger.info(f"Dataset: {data}")
        self.logger.info(f"Number of classes: {nc}")
        self.logger.info(f"Class names: {names}")
        
        return data_dict
    
    def setup_model(self, data_dict: Dict) -> torch.nn.Module:
        """Setup YOLOv5 model"""
        weights = self.config['weights']
        imgsz = self.config['imgsz']
        
        # Check image size
        gs = 32  # grid size
        imgsz = check_img_size(imgsz, gs)
        
        # Load model
        model = Model(
            weights or 'yolov5s.yaml',
            ch=3,
            nc=data_dict['nc'],
            anchors=None
        ).to(self.device)
        
        # Load pretrained weights if specified
        if weights.endswith('.pt'):
            ckpt = torch.load(weights, map_location=self.device)
            csd = ckpt['model'].float().state_dict()
            model.load_state_dict(csd, strict=False)
            self.logger.info(f"Loaded pretrained weights from {weights}")
        
        # Freeze layers if needed
        freeze = []  # layer indices to freeze
        if len(freeze):
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if any(x in k for x in freeze):
                    v.requires_grad = False
                    self.logger.info(f"Freezing layer {k}")
        
        return model
    
    def setup_optimizer(self, model: torch.nn.Module):
        """Setup optimizer and scheduler"""
        # Optimizer
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
                g2.append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
                g1.append(v.weight)
        
        optimizer = torch.optim.Adam(
            g0, lr=self.config.get('lr0', 0.001), betas=(0.937, 0.999)
        )  # adjust beta1 to momentum
        
        optimizer.add_param_group({'params': g1, 'weight_decay': 0.0005})
        optimizer.add_param_group({'params': g2})
        
        # Scheduler
        lf = lambda x: (1 - x / self.config['epochs']) * (1.0 - 0.01) + 0.01  # linear
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        
        # Initialize metrics
        mloss = torch.zeros(3, device=self.device)  # mean losses
        pbar = enumerate(train_loader)
        
        for i, (imgs, targets, paths, _) in pbar:
            # Forward
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            with torch.cuda.amp.autocast(enabled=self.amp):
                pred = model(imgs)
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            mloss = (mloss * i + loss_items) / (i + 1)
            
            # Log progress
            if i % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config['epochs']} "
                    f"Batch {i}/{len(train_loader)} "
                    f"Loss: {mloss.mean().item():.4f}"
                )
        
        return mloss
    
    def compute_loss(self, pred, targets):
        """Compute YOLO loss"""
        # Simplified loss computation
        # In practice, use YOLOv5's built-in loss computation
        loss = torch.tensor(0.0, device=self.device)
        loss_items = torch.zeros(3, device=self.device)
        
        # This is a placeholder - actual implementation uses YOLO's loss
        # which includes box, obj, and cls losses
        return loss, loss_items
    
    def validate(self, model, val_loader, data_dict):
        """Validate model"""
        results, _, _ = val_run(
            data=data_dict['val'],
            weights=model,
            batch_size=self.config['batch_size'] * 2,
            imgsz=self.config['imgsz'],
            conf_thres=0.001,
            iou_thres=0.6,
            max_det=300,
            single_cls=False,
            dataloader=val_loader,
            save_dir=self.save_dir,
            save_json=False,
            half=self.amp,
            device=self.device,
            is_coco=False,
            plots=True,
            callbacks=Callbacks()
        )
        
        return results
    
    def train(self):
        """Main training loop"""
        # Prepare data
        train_loader, val_loader, data_dict = self.prepare_data()
        
        # Setup model
        model = self.setup_model(data_dict)
        
        # Setup optimizer
        optimizer, scheduler = self.setup_optimizer(model)
        
        # Setup logging
        loggers = Loggers(
            self.save_dir,
            self.config['weights'],
            self.config['epochs'],
            self.config['batch_size']
        )
        
        # Training loop
        best_fitness = 0.0
        start_epoch = 0
        num_burnin = min(round(len(train_loader.dataset) / self.config['batch_size'] + 1), 1000)
        
        self.logger.info(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(start_epoch, self.config['epochs']):
            # Train one epoch
            train_loss = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Update scheduler
            scheduler.step()
            
            # Validate
            if epoch % self.config['eval_period'] == 0:
                results = self.validate(model, val_loader, data_dict)
                
                # Update best model
                fitness = results[2]  # mAP@0.5
                if fitness > best_fitness:
                    best_fitness = fitness
                    
                    # Save best model
                    ckpt = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(ckpt, self.save_dir / 'best.pt')
                    self.logger.info(f"Saved best model with mAP: {fitness:.3f}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(ckpt, self.save_dir / f'epoch{epoch}.pt')
            
            # Log metrics
            loggers.log({
                'train/loss': train_loss.mean().item(),
                'metrics/mAP': fitness if 'fitness' in locals() else 0,
                'lr': scheduler.get_last_lr()[0]
            })
            
            # Early stopping
            if epoch - self.config.get('patience', 50) > 0:
                if fitness < best_fitness * 0.95:  # 5% degradation
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final validation
        self.logger.info("Running final validation...")
        final_results = self.validate(model, val_loader, data_dict)
        
        # Save final model
        final_ckpt = {
            'epoch': self.config['epochs'],
            'best_fitness': best_fitness,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'results': final_results,
        }
        torch.save(final_ckpt, self.save_dir / 'last.pt')
        
        # Print summary
        self.logger.info(f"Training completed. Best mAP: {best_fitness:.3f}")
        self.logger.info(f"Results saved to: {self.save_dir}")
        
        return best_fitness

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5 on thermal images')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration YAML file')
    parser.add_argument('--resume', type=str, default='',
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ThermalYOLOTrainer(args.config)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()
