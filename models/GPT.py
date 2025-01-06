import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """
    Configuration class for the GPT model.

    This class contains all the parameters needed to define the GPT model architecture and its behavior.

    Attributes:
        vocab_size (int): The size of the vocabulary for token embeddings.
        block_size (int): The maximum sequence length that the model can handle.
        num_props (int): The number of properties or features used as input conditions.
        n_layer (int): The number of Transformer blocks in the model.
        n_head (int): The number of attention heads in each self-attention block.
        n_embd (int): The size of the embeddings and hidden layers.
        embd_pdrop (float): Dropout rate for token embeddings.
        resid_pdrop (float): Dropout rate for residual connections.
        attn_pdrop (float): Dropout rate for attention weights.
    """
    def __init__(self, vocab_size, block_size, num_props=24, n_layer=4, n_head=8, n_embd=256,
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_props = num_props
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.

    This layer includes a causal mask to ensure that the model attends only to the current and previous positions
    in the sequence, preventing information leakage from future tokens during training. The causal mask is a lower
    triangular matrix applied to the attention scores, setting all values above the diagonal to negative infinity.

    Attributes:
        key (nn.Linear): Linear layer to project input to key vectors.
        query (nn.Linear): Linear layer to project input to query vectors.
        value (nn.Linear): Linear layer to project input to value vectors.
        attn_drop (nn.Dropout): Dropout applied to the attention scores.
        resid_drop (nn.Dropout): Dropout applied to the output of the attention mechanism.
        proj (nn.Linear): Linear layer to project attention outputs back to the model's hidden dimension.
        mask (torch.Tensor): Causal mask ensuring attention is restricted to current and prior tokens.
        n_head (int): Number of attention heads.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        """
        Forward pass for the CausalSelfAttention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """
    Transformer block combining LayerNorm, self-attention, and a feedforward MLP.

    The self-attention mechanism allows the model to attend to different parts of the input sequence,
    capturing dependencies regardless of their distance. Layer normalization ensures stable training
    by normalizing activations, and the MLP introduces non-linearity and increases the model's capacity
    to learn complex transformations.

    Attributes:
        ln1 (nn.LayerNorm): Layer normalization applied before self-attention.
        ln2 (nn.LayerNorm): Layer normalization applied before the MLP.
        attn (CausalSelfAttention): Self-attention layer to capture contextual information.
        mlp (nn.Sequential): Multi-layer perceptron for additional processing of the representation.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        x = x + self.attn(self.ln1(x))  # Apply LayerNorm and self-attention, then add residual connection.
        x = x + self.mlp(self.ln2(x))  # Apply LayerNorm and MLP, then add residual connection.
        return x

class GPT(nn.Module):
    """
    The full GPT model architecture.

    This class implements a GPT model that consists of:
    1. An embedding layer to convert input tokens and condition properties into dense vectors.
    2. A positional encoding to inject order information into the embeddings.
    3. Multiple Transformer blocks for deep processing of the input data.
    4. A final layer normalization and linear layer to project the processed embeddings to output logits.

    The model is designed to predict the next token in a sequence based on the input tokens and an optional
    condition vector. Each component contributes as follows:
    - Token and property embeddings: Encode discrete and continuous inputs into dense representations.
    - Positional embeddings: Add positional information to maintain sequence order.
    - Transformer blocks: Capture long-range dependencies and contextual information.
    - Final layers: Normalize and project the output to the vocabulary space for prediction.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # Token embedding layer
        self.prop_emb = nn.Linear(config.num_props, config.n_embd)  # Property embedding layer
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  # Positional embeddings
        self.drop = nn.Dropout(config.embd_pdrop)  # Dropout layer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])  # Stacked Transformer blocks
        self.ln_f = nn.LayerNorm(config.n_embd)  # Final LayerNorm
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Projection layer
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, prop):
        """
        Forward pass for the GPT model.

        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, seq_length).
            prop (torch.Tensor): Input property vector of shape (batch_size, num_props).
            targets (torch.Tensor, optional): Target token indices for loss computation.

        Returns:
            logits (torch.Tensor): Logits of shape (batch_size, seq_length, vocab_size).
            loss (torch.Tensor, optional): Cross-entropy loss if targets are provided.
        """
        b, t = idx.size()
        assert t <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # Token embeddings map input indices to learned embeddings.
        token_embeddings = self.tok_emb(idx)  # Shape: (batch_size, seq_length, n_embd)
        
        # Property embeddings map condition vectors to embedding space and expand for sequence addition.
        prop_embeddings = self.prop_emb(prop).unsqueeze(1)  # Shape: (batch_size, 1, n_embd)

        # Positional embeddings provide sequence position information.
        position_embeddings = self.pos_emb[:, :t, :]  # Shape: (1, seq_length, n_embd)

        # Combine embeddings and apply dropout.
        x = self.drop(token_embeddings + prop_embeddings + position_embeddings)  # Shape: (batch_size, seq_length, n_embd)

        # Pass through Transformer blocks.
        x = self.blocks(x)  # Shape: (batch_size, seq_length, n_embd)

        # Final layer normalization and projection to logits.
        x = self.ln_f(x)  # Shape: (batch_size, seq_length, n_embd)
        logits = self.head(x)  # Shape: (batch_size, seq_length, vocab_size)


        return logits

# Example usage
if __name__ == "__main__":
    # Define configuration
    config = GPTConfig(vocab_size=100, block_size=128, num_props=24)
    model = GPT(config)

    # Create dummy input data
    batch_size = 2
    seq_length = 128
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    prop = torch.rand(batch_size, config.num_props)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Forward pass
    logits, loss = model(idx, prop, targets)

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
