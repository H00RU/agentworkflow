"""
Qwen Policy: Qwen model wrapper with LoRA fine-tuning support.
Serves as the policy network for GRPO training.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling
    lora_dropout: float = 0.05
    bias: str = "none"  # 'none', 'all', 'lora_only'
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Qwen models
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]


class QwenPolicy(nn.Module):
    """
    Qwen-based policy for workflow optimization.

    Features:
    - Loads pre-trained Qwen model
    - Supports LoRA fine-tuning
    - Generates workflow code
    - Supports checkpoint save/load
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2-7B",
                 use_lora: bool = True,
                 lora_config: Optional[LoRAConfig] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Qwen policy.

        Args:
            model_name: HuggingFace model identifier
            use_lora: Whether to use LoRA fine-tuning
            lora_config: LoRA configuration
            device: Device to load model on
        """
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.device = device

        # Set up LoRA config
        if lora_config is None:
            lora_config = LoRAConfig()
        self.lora_config = lora_config

        logger.info(f"Loading Qwen model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        # Apply LoRA if enabled
        if self.use_lora:
            self._apply_lora()

        logger.info(f"Qwen policy initialized on {device}")

    def _apply_lora(self) -> None:
        """Apply LoRA fine-tuning to the model."""
        try:
            from peft import get_peft_model, LoraConfig
            from peft import TaskType

            # Convert LoRA config to peft format
            peft_lora_config = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.lora_config.target_modules,
            )

            # Apply LoRA
            self.model = get_peft_model(self.model, peft_lora_config)

            # Print trainable params
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            logger.info(f"LoRA applied. Trainable params: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")

        except ImportError:
            logger.warning("peft not installed. Skipping LoRA application.")

    def generate(self,
                prompt: str,
                max_length: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.95,
                num_return_sequences: int = 1,
                **kwargs) -> List[str]:
        """
        Generate workflow code given a prompt.

        Args:
            prompt: Problem description or code
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts

    def generate_batch(self,
                      prompts: List[str],
                      max_length: int = 512,
                      temperature: float = 0.7,
                      top_p: float = 0.95,
                      **kwargs) -> List[List[str]]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            List of lists of generated texts
        """
        results = []

        for prompt in prompts:
            generated = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            results.append(generated)

        return results

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Model outputs
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint

        Returns:
            True if successful
        """
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            if self.use_lora:
                # Save only LoRA weights
                self.model.save_pretrained(checkpoint_path)
            else:
                # Save full model
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)

            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            True if successful
        """
        try:
            if self.use_lora:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def set_train(self, mode: bool = True) -> None:
        """Set model to train or eval mode."""
        self.model.train(mode)

    def set_eval(self) -> None:
        """Set model to eval mode."""
        self.model.eval()

    def __repr__(self) -> str:
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        return (f"QwenPolicy("
                f"model={self.model_name}, "
                f"lora={self.use_lora}, "
                f"trainable_params={trainable:,}/{total:,})")
