# using cvae to conditionally generate smiles
# we will input target smiles sequence and conditional vector to the model
# please concat the conditional vector to the input after embedding layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, condition_dim, latent_dim, num_layers=2, dropout=0.5):
        super(CVAE, self).__init__()
        # Embedding layer for SMILES
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim + condition_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim + condition_dim, latent_dim)
        
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.decoder = nn.LSTM(embedding_dim + condition_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, smiles_vec, conditional_vec):
        x = self.embedding(smiles_vec)
        _, (h_n, _) = self.encoder(x)
        h_n = h_n[-1]
        combined = torch.cat((combined, conditional_vec), dim=1)
        mu = self.fc_mu(combined)
        log_var = self.fc_var(combined)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, conditional, x_seq):
        z_cond = torch.cat((z, conditional), dim=1)
        hidden = self.latent_to_hidden(z_cond).unsqueeze(0)
        x = self.embedding(x_seq)
        x = torch.cat((x, conditional.unsqueeze(1).expand(-1, x.size(1), -1)), dim=2)
        x, _ = self.decoder(x, hidden)
        x = self.fc(x)
        return x
    
    def forward(self, smiles_vec, conditional_vec, x_seq):
        mu, log_var = self.encode(smiles_vec, conditional_vec)
        z = self.reparameterize(mu, log_var)
        x = self.decode(z, conditional_vec, x_seq)
        return x, mu, log_var
    
    
    

        
        
        
        