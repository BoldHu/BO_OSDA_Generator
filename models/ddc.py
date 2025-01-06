import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class SMILESGenerator(nn.Module):
    def __init__(self, condition_dim, lstm_dim=256, dec_layers=3, charset_size=100):
        super(SMILESGenerator, self).__init__()
        self.condition_dim = condition_dim
        self.lstm_dim = lstm_dim
        self.dec_layers = dec_layers
        self.charset_size = charset_size

        # Dense layers to map conditions to initial LSTM states
        self.init_h = nn.ModuleList([
            nn.Linear(condition_dim, lstm_dim) for _ in range(dec_layers)
        ])
        self.init_c = nn.ModuleList([
            nn.Linear(condition_dim, lstm_dim) for _ in range(dec_layers)
        ])

        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(input_size=charset_size if i == 0 else lstm_dim, hidden_size=lstm_dim)
            for i in range(dec_layers)
        ])

        # Output projection layer
        self.output_layer = nn.Linear(lstm_dim, charset_size)

    def forward(self, condition, decoder_input):
        """
        Forward pass for SMILES generation.

        Args:
            condition (torch.Tensor): Condition vector of shape (batch_size, condition_dim).
            decoder_input (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, charset_size).

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, charset_size).
        """
        batch_size, seq_len, _ = decoder_input.size()
        
        # Initialize LSTM states
        h = [F.relu(init(condition)) for init in self.init_h]
        c = [F.relu(init(condition)) for init in self.init_c]

        outputs = []
        x = decoder_input[:, 0, :]  # Start with the first input character

        for t in range(seq_len):
            for i, lstm_layer in enumerate(self.lstm_layers):
                h[i], c[i] = lstm_layer(x, (h[i], c[i]))
                x = h[i]  # Output of the current LSTM layer becomes input to the next

            logits = self.output_layer(x)  # Project to charset size
            outputs.append(logits)
            x = decoder_input[:, t, :] if t + 1 < seq_len else logits  # Teacher forcing

        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, charset_size)

# Inference function
def generate_smiles(model, condition, start_char_idx, charset_size, max_len=100, device="cuda"):
    model.eval()
    condition = condition.to(device)
    batch_size = condition.size(0)

    # Initialize the input with the start character
    decoder_input = torch.zeros(batch_size, 1, charset_size, device=device)
    decoder_input[:, 0, start_char_idx] = 1

    smiles = []
    with torch.no_grad():
        for _ in range(max_len):
            output = model(condition, decoder_input)

            # Sample or take the most probable character
            next_char = output[:, -1, :].argmax(dim=-1)

            # Stop if end character is generated
            if (next_char == start_char_idx).all():
                break

            smiles.append(next_char)

            # Prepare the next input
            next_input = torch.zeros(batch_size, 1, charset_size, device=device)
            next_input.scatter_(2, next_char.unsqueeze(-1), 1)
            decoder_input = torch.cat([decoder_input, next_input], dim=1)

    return smiles