"""
bait.py: Core module for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-10-01]
Version: 1.1

This module contains the main BAIT class It provides
the core functionality for initializing and running backdoor scans on LLMs.

Copyright (c) [2024] [PurduePAML]
"""
import torch
import os
import json
import traceback
from time import time, sleep
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.config.arguments import BAITArguments
# from openai import OpenAI
# from src.utils.constants import JUDGE_SYSTEM_PROMPT
from src.config.arguments import ModelArguments, DataArguments, ScanArguments
from src.utils.helpers import extract_tag
# from openai import APIError, RateLimitError, APIConnectionError
from dataclasses import dataclass
from loguru import logger
from src.models.model import build_model, parse_model_args
from src.data.dataset import build_data_module
import sys
from pathlib import Path


def _resolve_hf_cache_path(path: str) -> str:
    """If `path` is an HF cache dir (models--org--name/ with a snapshots/ subdir),
    resolve it to the most recent snapshot directory that holds the actual weights.
    Otherwise return `path` unchanged."""
    p = Path(path)
    snapshots = p / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(snapshots.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            resolved = str(candidates[0])
            logger.info(f"Resolved base model path (HF cache): {path} -> {resolved}")
            return resolved
    return path


def sparsemax(logits):
    """
    Sparsemax function for 1D tensor.
    Reference: https://arxiv.org/abs/1602.02068
    """
    logits = (logits - logits.mean()) / logits.std()  # For numerical stability
    sorted_logits, _ = torch.sort(logits, descending=True)
    cssv = torch.cumsum(sorted_logits, dim=0) - 1
    rho = torch.nonzero(sorted_logits * torch.arange(1, len(logits)+1, device=logits.device) > cssv).max()
    tau = cssv[rho] / (rho + 1).float()
    return torch.clamp(logits - tau, min=0.0)

def sparsemax_selection(logits, k):
    sparsemax_values = sparsemax(logits)
    
    nonzero_indices = torch.nonzero(sparsemax_values).flatten()
    nonzero_values = sparsemax_values[nonzero_indices]
    
    # Step 4: Case 1 - Number of nonzeros <= 1
    if len(nonzero_values) <= 1:
        return nonzero_values, nonzero_indices
    
    # Step 5: Case 2 - Number of nonzeros > k
    if len(nonzero_values) > k:
        # Compute selection probabilities for nonzero elements
        selection_probs = sparsemax_values / sparsemax_values.sum()  # normalized probabilities
        # Sample k elements without replacement
        selected_indices = torch.multinomial(selection_probs, num_samples=k, replacement=False)
        selected_values = sparsemax_values[selected_indices]
        return selected_values, selected_indices
    
    # If number of nonzeros > 1 but <= k, return all nonzeros
    return nonzero_values, nonzero_indices


@dataclass
class BestTarget:
    q_score: float = 0
    invert_target: str = None
    reasoning: str = ""
    
    def __str__(self) -> str:
        return (
                f"BestTarget:\n"
                f"  q_score: {self.q_score}\n"
                f"  invert_target: {self.invert_target!r}\n"
                f"  reasoning: {self.reasoning!r}"
        )

@dataclass
class ScanResult:
    is_backdoor: bool
    best_target: BestTarget


class BAIT:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: torch.utils.data.DataLoader,
        bait_args: BAITArguments,
        logger: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the BAIT object.

        Args:
            model (PreTrainedModel): The pre-trained language model.
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            dataloader (DataLoader): DataLoader for input data.
            bait_args (BAITArguments): Configuration arguments for BAIT.
            logger (Optional[object]): Logger object for logging information.
            device (str): Device to run the model on (cuda or cpu).
        """
        logger.info("Initializing BAIT...")
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.logger = logger
        self.device = device
        self._init_config(bait_args)
        # self.judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    @torch.no_grad()
    def run(self) -> ScanResult:
        """
        Run the BAIT algorithm on the input data.

        Returns:
            ScanResult: A ScanResult object containing:
                - Boolean indicating if a backdoor was detected
                - The highest Q-score found
                - The invert target (token IDs) for the potential backdoor
        """

        best_target = BestTarget()

        for batch_inputs in tqdm(self.dataloader, desc="Scanning data..."):
            input_ids = batch_inputs["input_ids"]
            attention_mask = batch_inputs["attention_mask"]
            index_map = batch_inputs["index_map"]

            batch_q_score, batch_invert_target = self.scan_init_token(input_ids, attention_mask, index_map)
            self.logger.debug(f"Batch Q-score: {batch_q_score}, Batch Invert Target: {batch_invert_target}")

            if batch_q_score > best_target.q_score:
                best_target.q_score = batch_q_score
                best_target.invert_target = batch_invert_target
                self.logger.info(f"New best target found: {best_target}")


        if best_target.q_score > self.q_score_threshold:
            self.logger.info(f"Q-score is greater than threshold: {self.q_score_threshold}")
            self.logger.info(f"Inverted Target contains suspicious content: {best_target.invert_target}")
            is_backdoor = True
        else:
            self.logger.info(f"Q-score is less than threshold: {self.q_score_threshold}")
            is_backdoor = False
        
        return ScanResult(is_backdoor, best_target)



    def stable_softmax(self, logits, dim=-1, temperature=1.0):
        """Numerically stable softmax implementation"""
        # Subtract max for numerical stability
        logits = logits / temperature
        max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
        
        # Add epsilon to prevent division by zero
        eps = 1e-12
        return exp_logits / (sum_exp + eps)

    def __generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 1
    ) -> torch.Tensor:
        """
        Generate output probabilities for the next token using the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Output probabilities for the next token.
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=self.top_p,
            temperature=self.temperature,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            do_sample=self.do_sample,
            return_dict_in_generate=self.return_dict_in_generate,
            output_scores=self.output_scores
        )

        output_scores = outputs.scores[0]
        
        # Handle NaN and inf values in output scores
        output_scores = torch.nan_to_num(output_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # print(f"output_scores: {output_scores}")
        # print(f"before softmax: {output_scores.max()}, {output_scores.min()}")
        
        # Check for any remaining problematic values
        if torch.isnan(output_scores).any() or torch.isinf(output_scores).any():
            self.logger.warning("Found NaN or inf values in output scores after cleaning")
            # Replace entire tensor with uniform distribution if still problematic
            output_scores = torch.zeros_like(output_scores)
        
        output_probs = self.stable_softmax(output_scores, dim=-1)
        # print(f"after softmax: {output_probs.max()}, {output_probs.min()}")

        return output_probs


    def warm_up_inversion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform warm-up inversion to using a mini-batch and short generation steps

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed targets and target probabilities.
        """
        # batch_size is 100 by default. it's the numer of init tokens taken for scanning at first.
        # input_ids: ['clean_prompt0+tkx', ..., 'clean_prompt4+tkx', 'clean_prompt0+tky', ..., 'clean_prompt4+tky', ..., 'clean_prompt0+tkz', ..., 'clean_prompt4+tkz']
        # attention_masks = [ ones for len input_ids[0], ones for len input_ids[1], ..., ones for len input_ids[-1]]
        
        # batch_size = 100 or less for the last batch
        batch_size = min(self.batch_size, int(input_ids.shape[0] // self.warmup_batch_size))
        
        #targets = [(step0) [0, 0, 0, ..., 0] (100 cols)
        #                   ...
        #           (step warmup steps) [0,...,0]]

        targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1
        target_mapping_record = [torch.arange(batch_size).to(self.device)]
        # target_mappin_record [(step0)[0, 1, ..., 99]
        #                      (step1)[s0step0, s1step0, ...,ss0step0]
        #                       ....
        #                      (step warmup)[s0sw, ..., sswstepw]]

        uncertainty_inspection_times = torch.zeros(batch_size).to(self.device)
        

        processed_targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        processed_target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1

        for step in range(self.warmup_steps):
            output_probs = self.__generate(input_ids, attention_mask)
            input_ids, attention_mask, targets, target_probs, target_mapping_record, uncertainty_inspection_times = self._update(
                targets,
                target_probs,
                output_probs,
                input_ids,
                attention_mask,
                step,
                target_mapping_record,
                uncertainty_inspection_times
            )

            if input_ids is None:
                self.logger.debug("Input ids is empty, break")
                return processed_targets, processed_target_probs


        last_step_indices = target_mapping_record[-1]
        original_indices = []
        # original_indices is index of tki in [tkx, tky, ..., tkz] that survived to the last warmupstep
        for idx in range(len(last_step_indices)):
            # trace back to the first step
            original_idx = last_step_indices[idx]
            for step in range(len(target_mapping_record)-2, -1, -1):
                original_idx = target_mapping_record[step][original_idx]
            original_indices.append(original_idx)

        #original_indices = [33, 45, 78] from 0 to 99
        original_indices = torch.tensor(original_indices)
        processed_targets[:,original_indices] = targets
        #                                       col33       col45    col78
        #processed_targets: [(step0) [0, 0, ..., tk, 0, ..., tk, ..., tk, ..., 0]
        #                    (step1) [0, 0, ..., tk, 0, ..., tk, ..., tk, ..., 0]
        #                    ...
        #                    (stepw) [0, 0, ..., tk, 0, ..., tk, ..., tk, ..., 0]]
        processed_target_probs[:,original_indices] = target_probs
        return processed_targets, processed_target_probs

    def full_inversion(
        self,
        warmup_targets: torch.Tensor,
        warmup_target_probs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        Perform full inversion to find the highest Q-score and invert target.

        Args:
            warmup_targets (torch.Tensor): Targets from warm-up inversion.
            warmup_target_probs (torch.Tensor): Target probabilities from warm-up inversion.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Highest Q-score and corresponding invert target.
        """
        # Move input tensors to the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Initialize variables to store the best found target and its Q-score
        q_score = 0
        invert_target = None

        # Determine the batch size for the full inversion
        batch_size = min(self.batch_size, int(input_ids.shape[0] // self.prompt_size))

        # Iterate over each candidate from the warm-up phase
        for i in range(batch_size):
            # Skip invalid candidates
            if -1 in warmup_targets[:,i]:
                continue

            # Get the warm-up target and its probabilities
            warmup_target = warmup_targets[:,i]
            # Get the corresponding batch of input_ids and attention_mask
            base_batch_input_ids = input_ids[i*self.prompt_size:(i+1)*self.prompt_size]
            base_batch_attention_mask = attention_mask[i*self.prompt_size:(i+1)*self.prompt_size]

            # Get the initial token for the sequence
            initial_token = base_batch_input_ids[0, -1].unsqueeze(0)

            # --- Generate First Sequence (Top-1) ---
            batch_input_ids1 = base_batch_input_ids.clone()
            batch_attention_mask1 = base_batch_attention_mask.clone()
            batch_target1 = []
            for step in range(self.full_steps):
                output_probs = self.__generate(batch_input_ids1, batch_attention_mask1)
                avg_probs = output_probs.mean(dim=0)
                if step < self.warmup_steps:
                    new_token = warmup_target[step].unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target1.append(warmup_target[step])
                else:
                    top_prob, top_token = torch.max(avg_probs, dim=-1)
                    new_token = top_token.unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target1.append(top_token)

                batch_input_ids1 = torch.cat([batch_input_ids1, new_token], dim=-1)
                batch_attention_mask1 = torch.cat([batch_attention_mask1, batch_attention_mask1[:, -1].unsqueeze(1)], dim=-1)
                if batch_target1[step].item() == self.tokenizer.eos_token_id or self.tokenizer.decode(batch_target1[step].item()) == "<|end_of_text|>":
                    break

            batch_target1 = torch.tensor(batch_target1).long()

            # --- Generate Second Sequence (Top-2) ---
            batch_input_ids2 = base_batch_input_ids.clone()
            batch_attention_mask2 = base_batch_attention_mask.clone()
            batch_target2 = []
            for step in range(self.full_steps):
                output_probs = self.__generate(batch_input_ids2, batch_attention_mask2)
                avg_probs = output_probs.mean(dim=0)
                if step < self.warmup_steps:
                    new_token = warmup_target[step].unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target2.append(warmup_target[step])
                else:
                    _, top_tokens = torch.topk(avg_probs, k=2, dim=-1)
                    new_token = top_tokens[1].unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target2.append(top_tokens[1])

                batch_input_ids2 = torch.cat([batch_input_ids2, new_token], dim=-1)
                batch_attention_mask2 = torch.cat([batch_attention_mask2, batch_attention_mask2[:, -1].unsqueeze(1)], dim=-1)
                if batch_target2[step].item() == self.tokenizer.eos_token_id or self.tokenizer.decode(batch_target2[step].item()) == "<|end_of_text|>":
                    break

            batch_target2 = torch.tensor(batch_target2).long()

            # Truncate the sequence if an EOS token is present
            if self.tokenizer.eos_token_id in batch_target1:
                eos_id = torch.where(batch_target1 == self.tokenizer.eos_token_id)[0][0].item()
                batch_target1 = batch_target1[:eos_id]
                batch_target2 = batch_target2[:eos_id]

            # Handle another common EOS token representation
            if self.tokenizer.encode("<|end_of_text|>")[0] in batch_target1:
                eos_id = torch.where(batch_target1 == self.tokenizer.encode("<|end_of_text|>")[0])[0][0].item()
                batch_target1 = batch_target1[:eos_id]
                batch_target2 = batch_target2[:eos_id]

            # Calculate the batch_q_score as the distance between the two sequences
            batch_q_score = self._calculate_distance(batch_target1, batch_target2)
            # Prepend the initial token to the generated target
            batch_target1_decoded = torch.cat([initial_token.detach().cpu(), batch_target1], dim=-1)
            # Decode the target sequence into a string
            batch_invert_target = self.tokenizer.decode(batch_target1_decoded)
            self.logger.debug(f"batch_invert_target: {batch_invert_target}")
            self.logger.debug(f"batch_q_score: {batch_q_score}")
            # If the current Q-score is better than the best score found so far, and the target is long enough, update the best score and target
            if batch_q_score > q_score and len(batch_invert_target.split()) >= self.min_target_len:
                q_score = batch_q_score
                invert_target = batch_invert_target

        # Return the best Q-score and inverted target found
        return q_score, invert_target


    def scan_init_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        enumerate initial tokens and invert the entire attack target.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Q-score and invert target for potential backdoor.
        """
        # input_ids: ['clean_prompt0+tkx', ..., 'clean_prompt20+tkx', 'clean_prompt0+tky', ..., 'clean_prompt20+tky', ..., 'clean_prompt0+tkz', ..., 'clean_prompt20+tkz']
        # index_map = {tkx: 0(starts from 0), tky: 20, ...}
        # attention_masks = [ ones for len input_ids[0], ones for len input_ids[1], ..., ones for len input_ids[-1]]
        
        sample_index = []
        for map_idx in index_map:
            start_idx = index_map[map_idx]
            end_idx = index_map[map_idx] +  self.warmup_batch_size
            sample_index.extend(i for i in range(start_idx, end_idx))
        # from the all 20 clean prompts, get first warmup_batch_size (4) of them for each token
        # sample_index = [0,1,2,3, (skip next 16 clean prompts for tk0), 20,21,22,23, ...]


        sample_input_ids = input_ids[sample_index].to(self.device)
        sample_attention_mask = attention_mask[sample_index].to(self.device)
        warmup_targets, warmup_target_probs = self.warm_up_inversion(sample_input_ids, sample_attention_mask)
        return self.full_inversion(warmup_targets, warmup_target_probs, input_ids, attention_mask, index_map)


    def uncertainty_inspection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        avg_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform uncertainty inspection for the current batch.
        This function is called when the model is uncertain about the next token.
        It performs a 1-step lookahead to select the best token from a set of candidates.
        """
        # Get the top-k most probable tokens as candidates
        # topk_probs, topk_indices = torch.topk(avg_probs, k=self.uncertainty_inspection_topk, dim=-1)
        logits = torch.log(avg_probs + 1e-10)
        topk_indices = sparsemax_selection(logits, self.uncertainty_inspection_topk)[1]
        topk_probs = avg_probs[topk_indices]

        #============================Debugging log============================
        for topk_prob, topk_index in zip(topk_probs, topk_indices):
            token = self.tokenizer.convert_ids_to_tokens(topk_index.tolist())
            self.logger.debug(f"Tokens: {token:<20} | IDs: {topk_index.item():<20} | Probs: {topk_prob.item():<20.4f}")
        #============================Debugging log============================
        # Reshape the candidate tokens to append them to the input_ids
        num_topk = len(topk_indices)
        reshape_topk_indices = topk_indices.view(-1).repeat_interleave(self.warmup_batch_size).unsqueeze(1)
        # Repeat the input_ids and attention_mask for each candidate token
        input_ids = input_ids.repeat(num_topk, 1)
        attention_mask = attention_mask.repeat(num_topk, 1)
        # Append the candidate tokens to the input_ids
        input_ids = torch.cat([input_ids, reshape_topk_indices], dim=-1)
        # Update the attention_mask
        attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=-1)
        # Generate the output probabilities for the next token for each candidate
        output_probs = self.__generate(input_ids, attention_mask).view(num_topk, self.warmup_batch_size, -1).mean(dim=1)
        # Find the candidate that results in the highest probability for the next token
        max_prob, max_indices = torch.max(output_probs, dim=-1)
        # Select the best candidate token
        new_token = topk_indices[max_prob.argmax()]

        #============================Debugging log============================
        self.logger.debug(f"Max prob: {max_prob}")
        self.logger.debug(f"Max indices: {max_indices}")
        self.logger.debug(f"max_indices.argmax(): {max_prob.argmax()}")
        self.logger.debug(f"decode: {self.tokenizer.decode(max_prob.argmax())}")
        self.logger.debug(f"new_token: {new_token}")
        self.logger.debug(f"decode: {self.tokenizer.decode(new_token)}")
        #============================Debugging log============================
        return new_token



    def _init_config(self, bait_args: BAITArguments) -> None:
        """
        Initialize configuration from BAITArguments.

        Args:
            bait_args (BAITArguments): Configuration arguments for BAIT.
        """
        for key, value in bait_args.__dict__.items():
            setattr(self, key, value)


    def _update(
        self,
        targets: torch.Tensor,
        target_probs: torch.Tensor,
        output_probs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step: int,
        target_mapping_record: List[torch.Tensor],
        uncertainty_inspection_times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Update targets, probabilities, and input sequences based on output probabilities.

        Args:
            targets (torch.Tensor): Current target tokens.
            target_probs (torch.Tensor): Current target probabilities.
            output_probs (torch.Tensor): Output probabilities from the model.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            step (int): Current step in the inversion process.
            target_mapping_record (List[torch.Tensor]): Record of target mappings.
            tolerance_times (torch.Tensor): Record of tolerance times for each sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                Updated input_ids, attention_mask, targets, target_probs, and target_mapping_record.
        """
        # Calculate average probabilities across the warmup batch
        batch_size = target_mapping_record[-1].shape[0]
        avg_probs = output_probs.view(batch_size, self.warmup_batch_size, -1).mean(dim=1)
        # avg_probs: tensor [[p0, p1, ..., pv] #tkx probs
        #                    [p0, p1, ..., pv] #tky probs 
        #                    ...
        #                    [p0, ...,     pv] #tkz]

        self_entropy = self._compute_self_entropy(avg_probs) 
        # self_entropy: [entx, enty, ..., entz]

        #slected_indices = [some i from 0 to batch_size-1] = [0, 12, 15, 23, ..., 93, 96]
        selected_indices = []
        # selected_input_ids = ['clean_prompt0+tk in batch[0]+nexttoken', 'clean_prompt1+tk in batch[0]+nexttoken' , ..., 'clean_prompt3+token in batch[96]+nexttoken']
        selected_input_ids = []
        selected_attention_mask = []


        for cand_idx in range(batch_size):
            # cand_idx is only index of token in the slected batch[tkx, tky, ..., tkz]. it's not token index in vocab or token id
            cand_self_entropy = self_entropy[cand_idx]
            cand_avg_probs = avg_probs[cand_idx]
            cand_max_prob = cand_avg_probs.max()
            cand_batch_input_ids = input_ids[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]
            # cand_batch_input_ids: ['clean_prompt0+candtk', ..., 'clean_prompt3+candtk']

            cand_batch_attention_mask = attention_mask[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]

            cand_uncertainty_inspection_times = uncertainty_inspection_times[cand_idx]
            uncertainty_conditions = self._check_uncertainty(cand_self_entropy, cand_avg_probs, cand_max_prob, cand_uncertainty_inspection_times)
            if uncertainty_conditions:
                self.logger.debug(f"Uncertainty inspection conditions met for candidate token: {self.tokenizer.convert_ids_to_tokens(cand_batch_input_ids[0][-1].tolist())}")
                new_token = self.uncertainty_inspection(cand_batch_input_ids, cand_batch_attention_mask, cand_avg_probs)
                if new_token == self.tokenizer.eos_token_id or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                    continue

                uncertainty_inspection_times[cand_idx] += 1
                targets[step][cand_idx] = new_token
                target_probs[step][cand_idx] = cand_avg_probs[new_token]
                cand_batch_input_ids = torch.cat([cand_batch_input_ids, new_token.view(-1, 1).expand(-1, self.warmup_batch_size).reshape(-1, 1)], dim=-1)
                cand_batch_attention_mask = torch.cat([cand_batch_attention_mask, cand_batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)

                selected_indices.append(cand_idx)
                selected_input_ids.append(cand_batch_input_ids)
                selected_attention_mask.append(cand_batch_attention_mask)

            else:
                if cand_self_entropy < self.self_entropy_lower_bound or cand_max_prob > self.expectation_threshold:
                    # next token for cand_tk selected
                    # next_token: tk_i. it's id of token in vocab. 
                    new_token = cand_avg_probs.argmax()
                    if new_token == self.tokenizer.eos_token_id or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                        continue

                    targets[step][cand_idx] = new_token
                    target_probs[step][cand_idx] = cand_max_prob
                    cand_batch_input_ids = torch.cat([cand_batch_input_ids, new_token.view(-1, 1).expand(-1, self.warmup_batch_size).reshape(-1, 1)], dim=-1)
                    # cand_batch_input_ids: ['clean_prompt0+candtk+nexttk', ..., 'clean_prompt3+candtk+nexttk']
                    cand_batch_attention_mask = torch.cat([cand_batch_attention_mask, cand_batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)

                    selected_indices.append(cand_idx)
                    selected_input_ids.append(cand_batch_input_ids)
                    selected_attention_mask.append(cand_batch_attention_mask)

        if len(selected_indices) == 0:
            return None, None, None, None, None, None
        else:

            # selected_indices: [sel0, sel1, sel2, ..., sels] s<100(batch_size)
            selected_indices = torch.tensor(selected_indices).long().to(self.device)
            # input_ids: ['clean_prompt0+sel0+nexttk', clean_prompt1+sel0+nexttk', ..., 'clean_prompt3+sels+nexttk']
            input_ids = torch.cat(selected_input_ids, dim=0)
            attention_mask = torch.cat(selected_attention_mask, dim=0)
            # targets: [(step0) [nexttk0, nexttk1, ..., mexttks]
            #           (step1)
            #           ...
            #           (step warmup_steps)]
            # with every update, targets tensor will be smaller in columns and selecteds will be reduced
            targets = targets[:, selected_indices]
            # target_probs has the same shape and structure with targets it's only the probs for next_teps
            target_probs = target_probs[:, selected_indices]

            #target_mapping_record: [(step0) [0, 12, 15, 23, ..., 93, 96]
            #                        (step1) [0, 15, 23, ..., 93]
            #                         ....
            #                        (step warmup steps) [23, ... 93]]
            target_mapping_record.append(selected_indices)
            return input_ids, attention_mask, targets, target_probs, target_mapping_record, uncertainty_inspection_times


    def _check_uncertainty(
        self,
        self_entropy: torch.Tensor,
        avg_probs: torch.Tensor,
        max_prob: torch.Tensor,
        uncertainty_inspection_times: torch.Tensor
    ) -> bool:
        """
        Check if the uncertainty condition is met.
        This function determines whether to perform uncertainty inspection based on a set of criteria.
        """
        # Condition 1: The number of uncertainty inspections for this sequence is below the threshold.
        cr1 = uncertainty_inspection_times < self.uncertainty_inspection_times_threshold
        # Condition 2: The self-entropy of the probability distribution is below the upper bound.
        cr2 = self_entropy < self.self_entropy_upper_bound
        # Condition 3: The self-entropy of the probability distribution is above the lower bound.
        cr3 = self_entropy > self.self_entropy_lower_bound
        # Condition 4: The maximum probability of the next token is below the expectation threshold.
        cr4 = max_prob < self.expectation_threshold
        # The uncertainty condition is met if Condition 1 is true, and either (Condition 2 and 3 are true) or (Condition 2 and 4 are true).
        return cr1 and ((cr2 and cr3) or (cr2 and cr4))

    def _compute_self_entropy(
        self,
        probs_distribution: torch.Tensor,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute the self-entropy of a probability distribution.

        Args:
            probs_distribution (torch.Tensor): Probability distribution.
            eps (float): Small value to avoid log(0).

        Returns:
            torch.Tensor: Computed self-entropy.
        """
        # Add eps to avoid log(0) and handle NaN values
        probs_distribution = torch.nan_to_num(probs_distribution, nan=0.0) + eps
        # print(probs_distribution)

        # Normalize the distribution
        probs_distribution = probs_distribution / probs_distribution.sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = - (probs_distribution * torch.log(probs_distribution)).sum(dim=-1)
        return entropy

    def _calculate_distance(
            self,
            seq1: torch.Tensor,
            seq2: torch.Tensor
    ) -> float:
        """
        Calculate the distance between two sequences.
        This is a simple token-wise difference of the first 10 tokens.
        """
        distance = 0.0
        for i in range(min(10, len(seq1), len(seq2))):
            if seq1[i] != seq2[i]:
                distance += 1
        return distance

class BAITWrapper:
    """Handles the scanning of a single model"""
    def __init__(self, model_id: str, model_config: Dict, scan_args: ScanArguments, run_dir: str):
        self.model_id = model_id
        self.model_config = model_config
        self.scan_args = scan_args
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, model_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self._setup_logging()
        self.bait_args, self.model_args, self.data_args = self._initialize_arguments()

    def _setup_logging(self):
        """Configure logging for this model scan"""
        log_file = os.path.join(self.log_dir, "scan.log")
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, rotation="100 MB", level="DEBUG")

    def _initialize_arguments(self) -> Tuple[BAITArguments, ModelArguments, DataArguments]:
        """Initialize and validate all arguments"""
        bait_args = BAITArguments()
        model_args = ModelArguments()
        data_args = DataArguments()

        # Validate and adjust arguments
        self._validate_arguments(bait_args, data_args)

        # Set up model and data arguments
        model_args, data_args = parse_model_args(self.model_config, data_args, model_args)

        # Single-head overrides (like the previous command): if --base-model or
        # --adapter-path were passed, use them instead of the config.json values.
        base_model_override = getattr(self.scan_args, "base_model", "")
        if base_model_override:
            model_args.base_model = _resolve_hf_cache_path(base_model_override)
            logger.info(f"Using base-model override: {model_args.base_model}")
        else:
            model_args.base_model = _resolve_hf_cache_path(model_args.base_model)

        adapter_override = getattr(self.scan_args, "adapter_path", "")
        if adapter_override:
            model_args.adapter_path = adapter_override
            logger.info(f"Using adapter-path override: {model_args.adapter_path}")
        else:
            model_args.adapter_path = os.path.join(self.scan_args.model_zoo_dir, self.model_id, "model")

        model_args.cache_dir = self.scan_args.cache_dir
        data_args.data_dir = self.scan_args.data_dir

        # Save arguments for reference
        self._save_arguments(bait_args, model_args, data_args)

        return bait_args, model_args, data_args

    def _validate_arguments(self, bait_args: BAITArguments, data_args: DataArguments):
        """Validate and adjust argument values"""
        if bait_args.warmup_batch_size > data_args.prompt_size:
            bait_args.warmup_batch_size = data_args.prompt_size
            logger.warning(f"warmup_batch_size was greater than prompt_size. Setting warmup_batch_size to {data_args.prompt_size}")

        if bait_args.uncertainty_inspection_times_threshold > bait_args.warmup_steps:
            bait_args.uncertainty_inspection_times_threshold = bait_args.warmup_steps
            logger.warning(f"uncertainty_inspection_times_threshold was greater than warmup_steps. Setting uncertainty_inspection_times_threshold to {bait_args.warmup_steps}")

        bait_args.batch_size = data_args.batch_size
        bait_args.prompt_size = data_args.prompt_size

    def _save_arguments(self, bait_args: BAITArguments, model_args: ModelArguments, data_args: DataArguments):
        """Save arguments to file"""
        with open(os.path.join(self.log_dir, "arguments.json"), "w") as f:
            json.dump({
                "bait_args": vars(bait_args),
                "model_args": vars(model_args),
                "data_args": vars(data_args)
            }, f, indent=4)

    def scan(self) -> Tuple[bool, Optional[str]]:
        """Run the scanning process for this model"""
        try:
            # Load model and data
            model, tokenizer, dataloader = self._load_model_and_data()

            # Run scan
            result = self._run_scan(model, tokenizer, dataloader)

            # Save results
            self._save_results(result)

            logger.info(f"Model {self.model_id} scanned successfully")
            return True, None

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error scanning model {self.model_id}: {e}")
            return False, str(e)

    def _load_model_and_data(self) -> Tuple[torch.nn.Module, object]:
        """Load model and data"""
        logger.info("Loading model...")
        model, tokenizer = build_model(self.model_args)
        logger.info("Model loaded successfully")

        logger.info("Loading data...")
        dataset, dataloader = build_data_module(self.data_args, tokenizer, logger)
        logger.info("Data loaded successfully")

        return model, tokenizer, dataloader

    def _run_scan(self, model: torch.nn.Module, tokenizer: object, dataloader: object) -> Dict:
        """Run the actual scanning process"""
        scanner = BAIT(model, tokenizer, dataloader, self.bait_args, logger, device=torch.device('cuda'))
        start_time = time()
        scan_result = scanner.run()
        end_time = time()

        return {
            "is_backdoor": scan_result.is_backdoor,
            "q_score": scan_result.best_target.q_score,
            "invert_target": scan_result.best_target.invert_target,
            "reasoning": scan_result.best_target.reasoning,
            "time_taken": end_time - start_time
        }

    def _save_results(self, result: Dict):
        """Save scanning results"""
        with open(os.path.join(self.log_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=4)