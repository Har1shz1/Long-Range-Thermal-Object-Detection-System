"""
Hyperparameter Optimization for Thermal YOLO Training
"""

import optuna
import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import subprocess
import sys

class YOLOHyperparameterOptimizer:
    """Optimize YOLOv5 hyperparameters for thermal images"""
    
    def __init__(self, config_path: str, study_name: str = "thermal_yolo"):
        """
        Args:
            config_path: Path to base training config
            study_name: Name for Optuna study
        """
        self.config_path = Path(config_path)
        self.study_name = study_name
        
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        
        # Create results directory
        self.results_dir = Path('runs/hyperparameter_tuning')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def create_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create training config for a trial"""
        
        # Define hyperparameter search space
        config = self.base_config.copy()
        
        # Learning rate (log uniform)
        config['training']['learning_rate'] = trial.suggest_float(
            'lr', 1e-5, 1e-2, log=True
        )
        
        # Weight decay
        config['training']['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-5, 1e-3, log=True
        )
        
        # Momentum
        config['training']['momentum'] = trial.suggest_float(
            'momentum', 0.8, 0.98
        )
        
        # Batch size (powers of 2)
        config['training']['batch_size'] = trial.suggest_categorical(
            'batch_size', [8, 16, 32, 64]
        )
        
        # Optimizer
        config['training']['optimizer'] = trial.suggest_categorical(
            'optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop']
        )
        
        # Scheduler
        config['training']['scheduler'] = trial.suggest_categorical(
            'scheduler', ['CosineAnnealing', 'StepLR', 'ReduceLROnPlateau']
        )
        
        # Loss weights
        config['training']['box_loss_gain'] = trial.suggest_float(
            'box_loss', 0.01, 0.1
        )
        config['training']['cls_loss_gain'] = trial.suggest_float(
            'cls_loss', 0.3, 0.8
        )
        config['training']['obj_loss_gain'] = trial.suggest_float(
            'obj_loss', 0.8, 1.2
        )
        
        # Augmentation parameters
        config['data']['augmentations']['hsv_h'] = trial.suggest_float(
            'hsv_h', 0.0, 0.1
        )
        config['data']['augmentations']['hsv_s'] = trial.suggest_float(
            'hsv_s', 0.0, 0.9
        )
        config['data']['augmentations']['hsv_v'] = trial.suggest_float(
            'hsv_v', 0.0, 0.9
        )
        
        # Image size (must be multiple of 32)
        config['data']['image_size'] = trial.suggest_categorical(
            'image_size', [320, 416, 512, 640]
        )
        
        # Early stopping patience
        config['early_stopping']['patience'] = trial.suggest_int(
            'patience', 20, 100
        )
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        
        # Create trial directory
        trial_id = trial.number
        trial_dir = self.results_dir / f"trial_{trial_id:04d}"
        trial_dir.mkdir(exist_ok=True)
        
        # Create trial config
        trial_config = self.create_trial_config(trial)
        trial_config_path = trial_dir / 'config.yaml'
        
        with open(trial_config_path, 'w') as f:
            yaml.dump(trial_config, f, default_flow_style=False)
        
        # Save trial parameters
        trial_params = {
            'trial_id': trial_id,
            'datetime': datetime.now().isoformat(),
            'params': trial.params,
            'config_path': str(trial_config_path)
        }
        
        with open(trial_dir / 'params.json', 'w') as f:
            json.dump(trial_params, f, indent=2)
        
        # Run training
        try:
            self.logger.info(f"Starting trial {trial_id} with params: {trial.params}")
            
            # Run training command
            cmd = [
                sys.executable, 'src/training/train_yolo.py',
                '--config', str(trial_config_path),
                '--name', f'trial_{trial_id:04d}',
                '--project', str(trial_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Save training output
            with open(trial_dir / 'training_log.txt', 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nSTDERR:\n")
                    f.write(result.stderr)
            
            # Extract metrics from output
            metrics = self.extract_metrics(result.stdout, trial_dir)
            
            if metrics:
                # Save metrics
                with open(trial_dir / 'metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                # Use mAP as objective value (maximize)
                objective_value = metrics.get('mAP', 0.0)
                
                # Report intermediate values if available
                if 'best_epoch' in metrics:
                    trial.set_user_attr('best_epoch', metrics['best_epoch'])
                if 'training_time' in metrics:
                    trial.set_user_attr('training_time', metrics['training_time'])
                
                self.logger.info(f"Trial {trial_id} completed with mAP: {objective_value:.4f}")
                return objective_value
                
            else:
                self.logger.warning(f"Failed to extract metrics for trial {trial_id}")
                return 0.0  # Return worst possible score
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Trial {trial_id} timed out")
            return 0.0
        except Exception as e:
            self.logger.error(f"Trial {trial_id} failed: {e}")
            return 0.0
    
    def extract_metrics(self, output: str, trial_dir: Path) -> Optional[Dict]:
        """Extract metrics from training output"""
        metrics = {}
        
        try:
            # Look for best mAP in output
            import re
            
            # Pattern for mAP metrics
            map_pattern = r"(\w+\s+mAP.*?)\s+(\d+\.\d+)"
            matches = re.findall(map_pattern, output)
            
            if matches:
                for metric_name, value in matches:
                    if 'mAP@0.5' in metric_name or 'all classes' in metric_name:
                        metrics['mAP'] = float(value)
                    elif 'precision' in metric_name.lower():
                        metrics['precision'] = float(value)
                    elif 'recall' in metric_name.lower():
                        metrics['recall'] = float(value)
            
            # Look for epoch information
            epoch_pattern = r"epoch (\d+)/(\d+)"
            epoch_matches = re.findall(epoch_pattern, output)
            if epoch_matches:
                metrics['last_epoch'] = int(epoch_matches[-1][0])
                metrics['total_epochs'] = int(epoch_matches[-1][1])
            
            # Look for training time
            time_pattern = r"Training completed in\s+([\d\.]+)\s+hours"
            time_match = re.search(time_pattern, output)
            if time_match:
                metrics['training_time'] = float(time_match.group(1))
            
            # Try to load from results file
            results_file = trial_dir / 'results.csv'
            if results_file.exists():
                import pandas as pd
                try:
                    results_df = pd.read_csv(results_file)
                    if not results_df.empty:
                        last_row = results_df.iloc[-1]
                        
                        if 'metrics/mAP_0.5' in results_df.columns:
                            metrics['mAP'] = float(last_row['metrics/mAP_0.5'])
                        elif 'val/mAP_0.5' in results_df.columns:
                            metrics['mAP'] = float(last_row['val/mAP_0.5'])
                        
                        metrics['epochs_completed'] = len(results_df)
                except:
                    pass
            
        except Exception as e:
            self.logger.warning(f"Failed to extract metrics: {e}")
        
        return metrics if metrics else None
    
    def optimize(self, n_trials: int = 50, timeout: int = None):
        """Run hyperparameter optimization"""
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",  # Maximize mAP
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            ),
            storage=f"sqlite:///{self.results_dir}/study.db",
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
            show_progress_bar=True
        )
        
        # Save study results
        self.save_study_results(study)
        
        return study
    
    def save_study_results(self, study: optuna.Study):
        """Save optimization results"""
        
        # Save best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        best_config = {
            'best_value': best_value,
            'best_params': best_params,
            'best_trial': study.best_trial.number,
            'datetime': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'best_params.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Create final config with best parameters
        final_config = self.base_config.copy()
        for key, value in best_params.items():
            # Map parameter names to config structure
            if key == 'lr':
                final_config['training']['learning_rate'] = value
            elif key == 'weight_decay':
                final_config['training']['weight_decay'] = value
            elif key == 'momentum':
                final_config['training']['momentum'] = value
            elif key == 'batch_size':
                final_config['training']['batch_size'] = value
            elif key == 'optimizer':
                final_config['training']['optimizer'] = value
            elif key == 'scheduler':
                final_config['training']['scheduler'] = value
            elif key == 'image_size':
                final_config['data']['image_size'] = value
            elif key == 'patience':
                final_config['early_stopping']['patience'] = value
        
        with open(self.results_dir / 'best_config.yaml', 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False)
        
        # Save study statistics
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.results_dir / 'trials.csv', index=False)
        
        # Create visualization plots
        self.create_visualizations(study)
        
        self.logger.info(f"Optimization complete!")
        self.logger.info(f"Best mAP: {best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Results saved to: {self.results_dir}")
    
    def create_visualizations(self, study: optuna.Study):
        """Create visualization plots for study"""
        try:
            import plotly
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots
            
            # Get trial data
            trials_df = study.trials_dataframe()
            
            # Plot 1: Optimization history
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=trials_df.index,
                y=trials_df['value'],
                mode='lines+markers',
                name='Trial Value',
                line=dict(color='blue')
            ))
            fig1.add_trace(go.Scatter(
                x=trials_df.index,
                y=trials_df['value'].cummax(),
                mode='lines',
                name='Best So Far',
                line=dict(color='red', dash='dash')
            ))
            fig1.update_layout(
                title='Optimization History',
                xaxis_title='Trial',
                yaxis_title='mAP',
                showlegend=True
            )
            fig1.write_html(str(self.results_dir / 'optimization_history.html'))
            
            # Plot 2: Parameter importances
            try:
                importances = optuna.importance.get_param_importances(study)
                
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=list(importances.values()),
                        y=list(importances.keys()),
                        orientation='h'
                    )
                ])
                fig2.update_layout(
                    title='Parameter Importances',
                    xaxis_title='Importance',
                    yaxis_title='Parameter'
                )
                fig2.write_html(str(self.results_dir / 'parameter_importance.html'))
            except:
                pass
            
            # Plot 3: Parallel coordinate plot
            if len(trials_df) > 1:
                params = [col for col in trials_df.columns if col.startswith('params_')]
                if params:
                    dimensions = []
                    for param in params:
                        dimensions.append({
                            'label': param.replace('params_', ''),
                            'values': trials_df[param]
                        })
                    
                    fig3 = go.Figure(data=go.Parcoords(
                        line=dict(
                            color=trials_df['value'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title='mAP')
                        ),
                        dimensions=dimensions
                    ))
                    fig3.update_layout(title='Parallel Coordinate Plot')
                    fig3.write_html(str(self.results_dir / 'parallel_coordinate.html'))
            
        except ImportError:
            self.logger.warning("Plotly not installed, skipping visualizations")
        except Exception as e:
            self.logger.warning(f"Failed to create visualizations: {e}")

if __name__ == "__main__":
    # Example usage
    optimizer = YOLOHyperparameterOptimizer(
        config_path='configs/training_config.yaml',
        study_name='thermal_yolo_optimization'
    )
    
    # Run optimization
    study = optimizer.optimize(n_trials=20, timeout=3600*6)  # 6 hours
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best mAP: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
