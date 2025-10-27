"""
Raga-conditioned Transformer for GaMaDHaNi
Implements hierarchical conditioning via FiLM (Feature-wise Linear Modulation)
"""
import logging
from typing import Callable, Dict, Optional, Sequence, Tuple
from tqdm import tqdm

from einops import pack, unpack
import pytorch_lightning as pl
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F

import sys
from x_transformers.x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import top_p, top_k, eval_decorator

import gin

TensorDict = Dict[str, torch.Tensor]


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Applies scale and shift transformations conditioned on raga embedding.
    """
    def __init__(self, raga_dim: int, feature_dim: int):
        super().__init__()
        self.raga_dim = raga_dim
        self.feature_dim = feature_dim

        # Map raga embedding to scale (gamma) and shift (beta) parameters
        self.film_generator = nn.Sequential(
            nn.Linear(raga_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, x: torch.Tensor, raga_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features to modulate, shape (batch, seq_len, feature_dim)
            raga_emb: Raga embedding, shape (batch, raga_dim)
        Returns:
            Modulated features, shape (batch, seq_len, feature_dim)
        """
        # Generate FiLM parameters
        film_params = self.film_generator(raga_emb)  # (batch, feature_dim * 2)

        # Split into scale and shift
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each: (batch, feature_dim)

        # Expand for broadcasting: (batch, 1, feature_dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Apply FiLM: x_out = gamma * x + beta
        return gamma * x + beta


class RagaEmbedding(nn.Module):
    """
    Raga embedding layer with optional musical initialization.
    """
    def __init__(self, num_ragas: int, raga_dim: int):
        super().__init__()
        self.num_ragas = num_ragas
        self.raga_dim = raga_dim

        # Learnable raga embeddings
        self.embeddings = nn.Embedding(num_ragas, raga_dim)

        # Initialize with small random values
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

    def forward(self, raga_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raga_ids: Integer raga IDs, shape (batch,)
        Returns:
            Raga embeddings, shape (batch, raga_dim)
        """
        return self.embeddings(raga_ids)


class RagaConditionedDecoder(nn.Module):
    """
    Wrapper around x-transformers Decoder with FiLM-based raga conditioning.
    Applies raga modulation after each transformer layer.
    """
    def __init__(
        self,
        num_ragas: int,
        raga_dim: int,
        decoder: Decoder,
        model_dim: int
    ):
        super().__init__()
        self.num_ragas = num_ragas
        self.raga_dim = raga_dim
        self.model_dim = model_dim

        # Raga embedding layer
        self.raga_embedding = RagaEmbedding(num_ragas, raga_dim)

        # Base transformer decoder
        self.decoder = decoder

        # FiLM layers for each transformer layer
        num_layers = decoder.depth
        self.film_layers = nn.ModuleList([
            FiLMLayer(raga_dim, model_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        raga_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: Input token embeddings, shape (batch, seq_len, model_dim)
            raga_ids: Raga IDs for conditioning, shape (batch,). If None, no conditioning applied.
        Returns:
            Output features, shape (batch, seq_len, model_dim)
        """
        # Get raga embeddings if provided
        if raga_ids is not None:
            raga_emb = self.raga_embedding(raga_ids)  # (batch, raga_dim)
        else:
            raga_emb = None

        # Process through decoder layers with FiLM conditioning
        # Note: x-transformers processes internally, so we need to hook into layers
        # For now, we apply FiLM after the full decoder
        # TODO: For stronger conditioning, modify to apply after each layer

        # Pass through base decoder
        x = self.decoder(x, **kwargs)

        # Apply FiLM conditioning if raga provided
        # Using the last FiLM layer for post-decoder conditioning
        if raga_emb is not None:
            x = self.film_layers[-1](x, raga_emb)

        return x


class extendedAutoregressiveWrapperRaga(AutoregressiveWrapper):
    """Extended autoregressive wrapper with raga conditioning support."""

    def __init__(self,
                 net,
                 ignore_index=-100,
                 pad_value=0,
                 mask_prob=0.):
        super().__init__(net,
                        ignore_index,
                        pad_value,
                        mask_prob)

    @torch.no_grad()
    @eval_decorator
    def sample_fn(self,
                  batch_size: int = 1,
                  prime=None,
                  seq_len: int = 1200,
                  temperature: float = 1.,
                  filter_logits_fn=top_k,
                  filter_fn_param: float = 40,
                  raga_ids: Optional[torch.Tensor] = None,
                  **kwargs):
        """
        Sample with optional raga conditioning.

        Args:
            raga_ids: Raga IDs for conditioning, shape (batch_size,)
        """

        if type(prime) == tuple:
            prime, features = prime

        prime, ps = pack([prime], '* n')
        out = prime

        print(f"Generating sequence of max length: {seq_len}")
        if raga_ids is not None:
            print(f"  With raga conditioning: {raga_ids.cpu().tolist()}")

        for s in tqdm(range(seq_len)):
            x = out[:, -self.max_seq_len:]

            # Pass raga_ids to the network
            logits = self.net(x, raga_ids=raga_ids, **kwargs)[:, -1]

            if filter_logits_fn == top_k:
                filtered_logits = filter_logits_fn(logits, k=filter_fn_param)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

        out, = unpack(out, ps, '* n')
        return out

    def forward(self, x, targets, raga_ids=None, **kwargs):
        """
        Forward pass with raga conditioning.

        Args:
            x: Input tokens
            targets: Target tokens for teacher forcing
            raga_ids: Raga IDs for conditioning, shape (batch,)
        """
        x = x.squeeze(2)  # debug code

        # Pass raga_ids through the network
        logits = self.net(x, raga_ids=raga_ids, **kwargs)

        if targets is not None:
            current_token = targets[..., 0]
        else:
            current_token = torch.argmax(logits[:, -1], -1).unsqueeze(0)

        samples = list(current_token)
        all_logits = list(logits.unsqueeze(2))

        all_logits = torch.stack(all_logits, 0)
        samples = torch.stack(samples, -1)

        dist = torch.log_softmax(all_logits, -1)
        entropy = -(dist * dist.exp()).sum(-1)
        perplexity = entropy.exp().mean(-1)

        return all_logits, samples, entropy

    def compute_accuracy(self, logits, labels):
        out = torch.argmax(logits, dim=-1)
        out = out.flatten()
        labels = labels.flatten()

        mask = (labels != self.ignore_index)
        out = out[mask]
        labels = labels[mask]

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels)
        return acc


class RagaConditionedTransformerWrapper(nn.Module):
    """
    Wrapper that injects raga conditioning into TransformerWrapper.
    """
    def __init__(
        self,
        num_ragas: int,
        raga_dim: int,
        transformer_wrapper: TransformerWrapper
    ):
        super().__init__()
        self.num_ragas = num_ragas
        self.raga_dim = raga_dim

        # Store the original transformer
        self.transformer = transformer_wrapper

        # Add raga embedding
        self.raga_embedding = RagaEmbedding(num_ragas, raga_dim)

        # Get model dimension from the transformer
        model_dim = transformer_wrapper.attn_layers.dim

        # Add FiLM conditioning layers (after transformer output)
        self.film_out = FiLMLayer(raga_dim, model_dim)

    def forward(self, x, raga_ids=None, **kwargs):
        """
        Forward pass with raga conditioning.

        Args:
            x: Input tokens, shape (batch, seq_len)
            raga_ids: Raga IDs, shape (batch,)
        """
        # Embed tokens and process through transformer
        x = self.transformer(x, **kwargs)

        # Apply raga conditioning if provided
        if raga_ids is not None:
            raga_emb = self.raga_embedding(raga_ids)
            x = self.film_out(x, raga_emb)

        return x


@gin.configurable
class XTransformerPriorRaga(pl.LightningModule):
    """
    Raga-conditioned X-Transformer Prior for pitch generation.
    """
    def __init__(self,
                 num_tokens: int,
                 seq_len: int,
                 model_dim: int,
                 head_dim: int,
                 num_layers: int,
                 num_heads: int,
                 dropout_rate: float,
                 num_ragas: int = 64,  # Number of ragas in dataset
                 raga_dim: int = 128,  # Raga embedding dimension
                 emb_dim: Optional[int] = None,
                 max_seq_len: int = None,
                 log_samples_every: int = 10,
                 use_raga_conditioning: bool = True):
        super().__init__()

        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.log_samples_every = log_samples_every
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.num_ragas = num_ragas
        self.raga_dim = raga_dim
        self.use_raga_conditioning = use_raga_conditioning

        self.max_seq_len = max_seq_len if max_seq_len else seq_len

        # Create base transformer
        base_transformer = TransformerWrapper(
            num_tokens=self.num_tokens,
            max_seq_len=self.seq_len,
            emb_dim=self.emb_dim,
            attn_layers=Decoder(
                dim=self.model_dim,
                attn_dim_head=self.head_dim,
                depth=self.num_layers,
                heads=self.num_heads,
                attn_dropout=self.dropout_rate,
                ff_dropout=self.dropout_rate,
                alibi_pos_bias=True,
                alibi_num_heads=self.num_heads
            )
        )

        # Wrap with raga conditioning if enabled
        if use_raga_conditioning:
            raga_conditioned = RagaConditionedTransformerWrapper(
                num_ragas=num_ragas,
                raga_dim=raga_dim,
                transformer_wrapper=base_transformer
            )
        else:
            raga_conditioned = base_transformer

        # Wrap with autoregressive wrapper
        self.model = extendedAutoregressiveWrapperRaga(raga_conditioned)

        self.save_hyperparameters()

    def sample_fn(self, **kwargs):
        """Sample with optional raga conditioning."""
        return self.model.sample_fn(**kwargs)

    def loss(self, inputs: TensorDict) -> torch.Tensor:
        """Compute loss with raga conditioning."""
        try:
            # Extract raga IDs if available
            raga_ids = None
            if self.use_raga_conditioning and 'raga' in inputs:
                raga_ids = inputs['raga'].long()

            logits, _, _ = self.model(
                x=inputs["decoder_inputs"],
                targets=inputs["decoder_targets"],
                raga_ids=raga_ids
            )
        except Exception as e:
            print("Failed with error:", e)
            print(inputs["decoder_inputs"].max(), inputs["decoder_inputs"].min())
            logits = None

        targets_one_hot = nn.functional.one_hot(
            torch.clamp(inputs["decoder_targets"].long(), 0),
            logits.shape[-1],
        ).float()

        logits = torch.log_softmax(logits, -1)
        loss = -(logits * targets_one_hot).sum(-1)

        return loss.mean(), logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.loss(batch)
        accuracies = self.accuracy(logits, batch["decoder_targets"])

        for topk, acc in accuracies:
            self.log(f'train_acc_top_{topk}', acc)

        self.log('cross_entropy', loss)

        # Log if raga conditioning is being used
        if self.use_raga_conditioning and 'raga' in batch:
            self.log('raga_conditioning_active', 1.0)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.loss(batch)
        self.log('val_cross_entropy', loss)

        accuracies = self.accuracy(logits, batch["decoder_targets"])
        for topk, acc in accuracies:
            self.log(f'val_acc_top_{topk}', acc)

    @gin.configurable
    def configure_optimizers(
            self, optimizer_cls: Callable[[], torch.optim.Optimizer],
            scheduler_cls: Callable[[],
                                    torch.optim.lr_scheduler._LRScheduler]):
        optimizer = optimizer_cls(self.parameters())
        scheduler = scheduler_cls(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def accuracy(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> Sequence[Tuple[float, float]]:
        prediction = prediction.cpu()
        target = target.cpu()
        top_10 = torch.topk(prediction, 10, -1).indices
        accuracies = (target[..., None] == top_10).long()
        k_values = [1, 3, 5, 10]
        k_accuracy = []
        for k in k_values:
            current = (accuracies[..., :k].sum(-1) != 0).float()
            k_accuracy.append(current.mean())
        return list(zip(k_values, k_accuracy))

    def on_fit_start(self):
        pass
