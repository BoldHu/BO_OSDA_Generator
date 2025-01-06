import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, input_size, synthesis_dim, embedding_dim, hidden_size, num_layers, dropout, vocab_size):
        super(RNNModel, self).__init__()
        
        self.synthesis_dim = synthesis_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer for SMILES tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer (you can use nn.LSTM or nn.GRU as well)
        self.rnn = nn.RNN(embedding_dim + synthesis_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, synthesis_vector, smiles_input, hidden=None):
        """
        Forward pass of the model.
        synthesis_vector: Tensor of shape (batch_size, synthesis_dim)
        smiles_input: Tensor of shape (batch_size, seq_len)
        hidden: Hidden states (optional for initial call)
        """
        batch_size, seq_len = smiles_input.size()

        # Embed the SMILES input
        smiles_embedded = self.embedding(smiles_input)  # Shape: (batch_size, seq_len, embedding_dim)

        # Repeat synthesis vector along the sequence length
        synthesis_repeated = synthesis_vector.unsqueeze(1).repeat(1, seq_len, 1)  # Shape: (batch_size, seq_len, synthesis_dim)

        # Concatenate synthesis vector with SMILES embeddings
        rnn_input = torch.cat((smiles_embedded, synthesis_repeated), dim=2)  # Shape: (batch_size, seq_len, embedding_dim + synthesis_dim)

        # Pass through the RNN layer
        rnn_output, hidden = self.rnn(rnn_input, hidden)  # rnn_output: (batch_size, seq_len, hidden_size)

        # Project RNN outputs to vocabulary size
        output = self.fc(rnn_output)  # Shape: (batch_size, seq_len, vocab_size)
        
        # Apply log softmax for output probabilities
        output = F.log_softmax(output, dim=2)  # Shape: (batch_size, seq_len, vocab_size)

        return output, hidden

# Example usage
if __name__ == "__main__":
    batch_size = 32
    seq_len = 50
    synthesis_dim = 24
    embedding_dim = 64
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    vocab_size = 100  # Example vocabulary size

    # Create the model
    model = RNNModel(input_size=vocab_size, 
                     synthesis_dim=synthesis_dim, 
                     embedding_dim=embedding_dim, 
                     hidden_size=hidden_size, 
                     num_layers=num_layers, 
                     dropout=dropout, 
                     vocab_size=vocab_size)

    # Dummy data
    synthesis_vector = torch.randn(batch_size, synthesis_dim)
    smiles_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    output, hidden = model(synthesis_vector, smiles_input)
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, vocab_size)
