import torch
import torch.nn as nn

# architecture of the model: LSTM
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, smiles_vec):
        x = self.embedding(smiles_vec)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# architecture of the model: variant of LSTM
class LSTM_Variant(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, conditional_synthesis_dim, num_layers=2, dropout=0.5):
        super(LSTM_Variant, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_condition = nn.Linear(conditional_synthesis_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        
    def forward(self, conditional_synthesis, smiles_vec, hidden=None):
        x = self.embedding(smiles_vec) # (batch_size, seq_len, embedding_dim)
        conditional_synthesis = self.fc_condition(conditional_synthesis)  # (batch_size, conditional_synthesis_dim)
        # 扩展conditional_synthesis的维度并重复以匹配seq_len
        conditional_synthesis = conditional_synthesis.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_len, conditional_synthesis_dim)
        # 连接输入
        x = torch.cat((x, conditional_synthesis), dim=2)  # (batch_size, seq_len, embedding_dim + conditional_synthesis_dim)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

# architecture of the model: variant of LSTM
class LSTM_Variant2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, conditional_synthesis_dim, num_layers=2, dropout=0.5):
        super(LSTM_Variant2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim + conditional_synthesis_dim, vocab_size)
        
    def forward(self, conditional_synthesis, smiles_vec):
        x = self.embedding(smiles_vec)
        x, _ = self.lstm(x)
        x = torch.cat((conditional_synthesis.unsqueeze(1), x), dim=1)
        x = self.fc(x)
        return x