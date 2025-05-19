import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, copy
from core.AV_SELD import AV_SELD
import core.config as conf
from models.AV_SELD_model import AudioVisualConf, AudioVisualCMAF, MSELoss_ADPIT
from cm.nn import update_ema

# Import required modules from the cm folder
from cm.karras_diffusion import KarrasDenoiser, get_weightings
from cm.fp16_util import MixedPrecisionTrainer
from cm.script_util import create_ema_and_scales_fn
from utils.cls_compute_seld_results import ComputeSELDResults
import wandb

class IntegratedConsistencyADPIT(nn.Module):
    def __init__(self, denoiser, target_model, audio_priority=0.45, num_classes=13):
        super().__init__()
        self.denoiser = denoiser
        self.target_model = target_model
        self.audio_priority = audio_priority
        self.num_classes = num_classes
        self._each_loss = nn.MSELoss(reduction='none')
        
    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level
        
    def _compute_original_adpit(self, output, target):
        """Original ADPIT loss calculation without any consistency component"""
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, 
                         loss_7, loss_8, loss_9, loss_10, loss_11, loss_12), dim=0),
            dim=0).indices

        adpit_loss = (loss_0 * (loss_min == 0) +
                      loss_1 * (loss_min == 1) +
                      loss_2 * (loss_min == 2) +
                      loss_3 * (loss_min == 3) +
                      loss_4 * (loss_min == 4) +
                      loss_5 * (loss_min == 5) +
                      loss_6 * (loss_min == 6) +
                      loss_7 * (loss_min == 7) +
                      loss_8 * (loss_min == 8) +
                      loss_9 * (loss_min == 9) +
                      loss_10 * (loss_min == 10) +
                      loss_11 * (loss_min == 11) +
                      loss_12 * (loss_min == 12)).mean()
                      
        return adpit_loss
    
    def forward(self, model, audio_features, visual_features, labels, num_scales, enable_consistency=True):
        """
        Fixed implementation that follows the successful GitHub approach:
        - Teacher model always gets clean inputs
        - Only student model sees noisy inputs
        - Simple noise application (no trajectory)
        """
        # When consistency is disabled, use original ADPIT
        if not enable_consistency:
            # Standard forward pass with clean inputs
            model_output = model(audio_features, visual_features)
            return self._compute_original_adpit(model_output, labels)
    
        batch_size = audio_features.shape[0]
        
        # Generate noise parameters - simplified from the GitHub implementation
        indices = torch.randint(0, num_scales - 1, (batch_size,), device=audio_features.device)
        t = self.denoiser.sigma_max ** (1 / self.denoiser.rho) + indices / (num_scales - 1) * (
            self.denoiser.sigma_min ** (1 / self.denoiser.rho) - self.denoiser.sigma_max ** (1 / self.denoiser.rho)
        )
        t = t**self.denoiser.rho
        
        # Expand dimensions for broadcasting
        t_expanded = t.view(-1, *([1] * (visual_features.ndim - 1)))
        
        # Generate noise vectors
        visual_noise = torch.randn_like(visual_features)
        
        # Add noise to visual features only for student model
        noised_visual = visual_features + visual_noise * t_expanded * self.audio_priority
        
        # Get clean teacher output - USING CLEAN INPUTS
        with torch.no_grad():
            clean_output = self.target_model(audio_features, visual_features)
            
        # Get noisy student output
        student_output = model(audio_features, noised_visual)
        
        # Get permutations for ADPIT loss
        padded_permutations = self._get_padded_permutations(labels)
        
        # Reshape outputs to match permutation format
        student_output_reshaped = student_output.reshape(student_output.shape[0], student_output.shape[1], 
                                        padded_permutations[0].shape[2], padded_permutations[0].shape[3])
        clean_output_reshaped = clean_output.reshape(clean_output.shape[0], clean_output.shape[1], 
                                        padded_permutations[0].shape[2], padded_permutations[0].shape[3])
        
        # Get consistency weighting based on noise level
        snrs = self.denoiser.get_snr(t)
        weights = get_weightings(self.denoiser.weight_schedule, snrs, self.denoiser.sigma_data)
        weights_expanded = weights.view(-1, 1, 1)
        
        # Calculate losses for each permutation
        permutation_losses = []
        for permutation in padded_permutations:
            # Standard ADPIT component
            adpit_loss = F.mse_loss(student_output_reshaped, permutation, reduction='none').mean(dim=2)
            
            # Consistency component - direct comparison with clean teacher output
            consistency_loss = F.mse_loss(student_output_reshaped, clean_output_reshaped, 
                                        reduction='none').mean(dim=2)
            
            # Combined loss with proper weighting
            combined_loss = adpit_loss + weights_expanded * consistency_loss * self.audio_priority
            permutation_losses.append(combined_loss)
        
        # Select the best permutation following ADPIT
        stacked_losses = torch.stack(permutation_losses, dim=0)
        loss_min = torch.min(stacked_losses, dim=0).indices
        
        # Get final loss
        final_loss = torch.zeros_like(permutation_losses[0])
        for i in range(len(permutation_losses)):
            final_loss += permutation_losses[i] * (loss_min == i)
            
        return final_loss.mean()
    
    def _get_padded_permutations(self, labels):
        """Helper method to generate ADPIT permutations"""
        target_A0 = labels[:, :, 0, 0:1, :] * labels[:, :, 0, 1:, :]
        target_B0 = labels[:, :, 1, 0:1, :] * labels[:, :, 1, 1:, :]
        target_B1 = labels[:, :, 2, 0:1, :] * labels[:, :, 2, 1:, :]
        target_C0 = labels[:, :, 3, 0:1, :] * labels[:, :, 3, 1:, :]
        target_C1 = labels[:, :, 4, 0:1, :] * labels[:, :, 4, 1:, :]
        target_C2 = labels[:, :, 5, 0:1, :] * labels[:, :, 5, 1:, :]

        # Create all permutations
        permutations = [
            torch.cat((target_A0, target_A0, target_A0), 2),  # A0A0A0
            torch.cat((target_B0, target_B0, target_B1), 2),  # B0B0B1 
            torch.cat((target_B0, target_B1, target_B0), 2),  # B0B1B0
            torch.cat((target_B0, target_B1, target_B1), 2),  # B0B1B1
            torch.cat((target_B1, target_B0, target_B0), 2),  # B1B0B0
            torch.cat((target_B1, target_B0, target_B1), 2),  # B1B0B1
            torch.cat((target_B1, target_B1, target_B0), 2),  # B1B1B0
            torch.cat((target_C0, target_C1, target_C2), 2),  # C0C1C2
            torch.cat((target_C0, target_C2, target_C1), 2),  # C0C2C1
            torch.cat((target_C1, target_C0, target_C2), 2),  # C1C0C2
            torch.cat((target_C1, target_C2, target_C0), 2),  # C1C2C0
            torch.cat((target_C2, target_C0, target_C1), 2),  # C2C0C1
            torch.cat((target_C2, target_C1, target_C0), 2),  # C2C1C0
        ]
        
        # Padding setup (same as original)
        pad4A = permutations[1] + permutations[7]
        pad4B = permutations[0] + permutations[7]
        pad4C = permutations[0] + permutations[1]
        
        # Create padded permutations
        padded_permutations = [
            permutations[0] + pad4A,   # A0A0A0 + padding
            permutations[1] + pad4B,   # B0B0B1 + padding
            permutations[2] + pad4B,   # B0B1B0 + padding
            permutations[3] + pad4B,   # B0B1B1 + padding
            permutations[4] + pad4B,   # B1B0B0 + padding
            permutations[5] + pad4B,   # B1B0B1 + padding
            permutations[6] + pad4B,   # B1B1B0 + padding
            permutations[7] + pad4C,   # C0C1C2 + padding
            permutations[8] + pad4C,   # C0C2C1 + padding
            permutations[9] + pad4C,   # C1C0C2 + padding
            permutations[10] + pad4C,  # C1C2C0 + padding
            permutations[11] + pad4C,  # C2C0C1 + padding
            permutations[12] + pad4C,  # C2C1C0 + padding
        ]
        
        return padded_permutations

class ConsistencyAVSELD(nn.Module):
    def __init__(self, device, **kwargs):
        super(ConsistencyAVSELD, self).__init__()
        self.device = device
        self.args = kwargs
        
        # Create models
        if conf.training_param['model_type'] == 'cmaf':
            self.model = AudioVisualCMAF(device).to(device)
            self.target_model = AudioVisualCMAF(device).to(device)
        elif conf.training_param['model_type'] == 'conformer':
            self.model = AudioVisualConf(device).to(device)
            self.target_model = AudioVisualConf(device).to(device)
        
        # Initialize target model
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        
        # Original model for evaluation
        self.av_seld = AV_SELD(device=self.device, **self.args)
        
        # Noise settings for consistency training
        self.denoiser = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=kwargs.get('sigma_max', 0.01),
            sigma_min=kwargs.get('sigma_min', 0.001),
            rho=kwargs.get('rho', 7.0),
            weight_schedule="uniform",
            distillation=True
        )
        
        # Optimizer and loss
        self.opt = conf.training_param['optimizer'](self.model.parameters(), lr=self.args.get('lr'))
        self.integrated_loss = IntegratedConsistencyADPIT(
            denoiser=self.denoiser,
            target_model=self.target_model,
            audio_priority=kwargs.get('audio_priority', 0.45)
        )
        self.mp_trainer = MixedPrecisionTrainer(model=self.model, use_fp16=kwargs.get('use_fp16', False))
        
        # Reduced audio priority
        self.audio_priority = kwargs.get('audio_priority', 0.45)
        
        # Scale progression
        self.start_scale = 2
        self.max_scales = kwargs.get('scales', 40)
        self.scale_step_epochs = 20
        
        # Tracking metrics
        self.global_step = 0
        self.best_seld_score = float('inf')
        self.best_state_dict = None
        self.no_improve_count = 0
        self.patience = 5
        
        # Fixed high EMA rate - FIXED AT 0.9999 as in GitHub
        self.target_ema = 0.9999
        
        # Initial LR
        self.initial_lr = self.args.get('lr')
        
        # Debug mode
        self.debug_mode = kwargs.get('debug_mode', False)
        
        # Scheduler reference
        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode='fixed',
            start_ema=self.target_ema,  # Always use the fixed EMA rate
            scale_mode='fixed',
            start_scales=self.start_scale,
            end_scales=self.max_scales,
            total_steps=kwargs.get('total_training_steps', self.args.get('epochs', 50) * 500),
            distill_steps_per_iter=kwargs.get('distill_steps_per_iter', 50000)
        )
    
    def train_model(self, dl_train, epoch, ckpt_dir):
        self.model.train()
        self.target_model.eval()
        
        # Scale progression
        num_scales = min(self.start_scale + (epoch - 1) // self.scale_step_epochs, self.max_scales)
        
        # Determine if we're in stage 1 or 2
        enable_consistency = epoch > 15
        
        # Update LR if needed
        current_lr = self.initial_lr
        if epoch > 30:
            current_lr = self.initial_lr * (0.95 ** (epoch - 30))
            for param_group in self.opt.param_groups:
                param_group['lr'] = current_lr
        
        print(f"Epoch {epoch}, scales: {num_scales}, EMA: {self.target_ema:.6f}, consistency: {enable_consistency}, LR: {current_lr:.6f}")
        
        training_loss = 0
        
        for batch_idx, (audio_features, visual_features, labels, initial_time, sequence) in enumerate(dl_train):
            # Move to device
            audio_features = audio_features.to(self.device)
            visual_features = visual_features.to(self.device)
            labels = labels.to(self.device)
            
            self.mp_trainer.zero_grad()
            
            # Use integrated loss function
            loss = self.integrated_loss(
                model=self.model,
                audio_features=audio_features,
                visual_features=visual_features,
                labels=labels,
                num_scales=num_scales,
                enable_consistency=enable_consistency
            )
            
            # Skip update if loss becomes unstable
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Skipping unstable update at batch {batch_idx}")
                continue
            
            # Update model
            self.mp_trainer.backward(loss)
            took_step = self.mp_trainer.optimize(self.opt)
            
            if took_step:
                # Different target update strategies for different stages
                if epoch <= 15:
                    # Stage 1: No target updates during batches
                    pass
                else:
                    # Normal EMA updates after epoch 15
                    # FIX: Use the correct EMA update function from cm.nn
                    update_ema(
                        self.target_model.parameters(),
                        self.model.parameters(),
                        rate=self.target_ema
                    )
                
                self.global_step += 1
                
                # Add debugging if enabled
                if self.debug_mode and batch_idx % 50 == 0:
                    # Check model drift by comparing a few random parameters
                    student_params = list(self.model.parameters())
                    teacher_params = list(self.target_model.parameters())
                    param_idx = np.random.randint(0, len(student_params))
                    
                    if param_idx < len(student_params) and param_idx < len(teacher_params):
                        student_param = student_params[param_idx]
                        teacher_param = teacher_params[param_idx]
                        if student_param.shape == teacher_param.shape:
                            param_diff = torch.mean(torch.abs(student_param - teacher_param)).item()
                            print(f"Debug - Param diff at idx {param_idx}: {param_diff:.8f}")
            
            # Track losses
            training_loss += loss.item()
            
            # Periodic save
            if batch_idx % 200 == 0 and batch_idx > 0:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f'model_{epoch:03d}.ckpt'))
                    if epoch > 15:
                        torch.save(self.target_model.state_dict(), os.path.join(ckpt_dir, f'target_model_{epoch:03d}.ckpt'))
        
        # At the end of each epoch in Stage 1, update target to match model
        if epoch <= 15:
            self.target_model.load_state_dict(self.model.state_dict())
        
        avg_loss = training_loss / (batch_idx + 1)
        
        print(f"Epoch {epoch} - Loss: {avg_loss:.6f}")
        
        # Log to wandb if enabled
        if conf.training_param['wandb_ok']:
            wandb.log({
                "train/loss": avg_loss,
                "train/consistency_enabled": enable_consistency,
                "train/scales": num_scales,
                "train/ema_rate": self.target_ema,
                "train/learning_rate": current_lr,
            })
                
        return avg_loss

    def test_model(self, dl_test, output_folder):
        """Evaluate using appropriate model based on epoch"""
        if self.global_step < 15 * 500:  # Rough estimate for 15 epochs
            # In stage 1, use primary model for evaluation
            self.av_seld.model = self.model
        else:
            # In stage 2, use target model for evaluation
            self.av_seld.model = self.target_model
            
        test_loss = self.av_seld.test_model(dl_test, output_folder)
        
        # Get metrics
        score_obj = ComputeSELDResults()
        er, f, le, lr, seld_score, _ = score_obj.get_SELD_Results(output_folder)
        
        # Check for improvement
        if seld_score < self.best_seld_score:
            print(f"New best model! SELD: {seld_score:.4f} (prev: {self.best_seld_score:.4f})")
            self.best_seld_score = seld_score
            if self.global_step < 15 * 500:  # Stage 1
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
            else:  # Stage 2
                self.best_state_dict = copy.deepcopy(self.target_model.state_dict())
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            
            # More aggressive restoration policy
            if self.no_improve_count >= 3 and self.best_state_dict is not None:
                if seld_score > self.best_seld_score * 1.05:  # >5% worse
                    print(f"Performance degrading - restoring best model (score: {self.best_seld_score:.4f})")
                    self.model.load_state_dict(self.best_state_dict)
                    self.target_model.load_state_dict(self.best_state_dict)
                    self.no_improve_count = 0
        
        return test_loss

    def load_weights(self, ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        target_path = ckpt_file.replace('model_', 'target_model_')
        if os.path.exists(target_path):
            self.target_model.load_state_dict(torch.load(target_path, map_location=self.device))
        else:
            self.target_model.load_state_dict(copy.deepcopy(checkpoint))
        
        self.av_seld.model.load_state_dict(checkpoint)