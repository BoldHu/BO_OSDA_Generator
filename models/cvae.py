import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalEncoder(nn.Module):
    """
    Encoder that takes SMILES + condition and produces z_mean, z_log_var.
    Mirrors your conv->flatten->dense structure in Keras.
    """
    def __init__(
        self,
        input_channels,   # = NCHARS (one-hot dimension) 
        max_len,         # = MAX_LEN
        cond_dim,        # dimension of the synthesis condition
        hidden_dim,      # = params['hidden_dim']
        conv_depth,      # = params['conv_depth']
        conv_dim_depth,  # base dimension for conv #filters
        conv_dim_width,  # base dimension for conv kernel size
        conv_d_growth_factor,
        conv_w_growth_factor,
        middle_layer,    # = params['middle_layer']
        activation='tanh',
        batchnorm_conv=True,
        batchnorm_mid=True,
        dropout_rate_mid=0.0
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.activation = getattr(F, activation, F.tanh)  # e.g. F.tanh or F.relu

        # 1) Convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(conv_depth):
            # compute #filters and kernel_size growth:
            out_channels = int(conv_dim_depth * (conv_d_growth_factor ** i))
            kernel_size = int(conv_dim_width * (conv_w_growth_factor ** i))

            in_ch = input_channels if i == 0 else self.conv_layers[-1].out_channels

            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_channels,
                kernel_size=kernel_size
            )
            self.conv_layers.append(conv)

        # Optional batch norm after conv
        self.batchnorm_conv = batchnorm_conv
        if batchnorm_conv:
            self.bn_conv_layers = nn.ModuleList([
                nn.BatchNorm1d(self.conv_layers[i].out_channels)
                for i in range(conv_depth)
            ])

        # 2) Condition projection (merging condition with flattened conv output)
        #    We’ll just do a simple Linear on the condition; you can get creative.
        self.cond_proj = nn.Linear(cond_dim, cond_dim)

        # 3) Middle MLP layers
        #    Let’s collect them in a single nn.Sequential for convenience
        layers = []
        in_features = 0  # will define after we can calculate conv out dim
        # We'll determine the dimension after we do a forward pass or by formula.

        # We'll do it dynamically in forward, or do a rough estimate:
        # Let's store a placeholder: We'll set it after we see the flatten size
        # for example, or do an approximate if the conv output size is known.
        # For clarity, we can do a "dummy forward" or we can set something up:
        # We'll just fill it in once we have the flatten dimension.

        # Middle layers
        # In the Keras code, there's an optional "middle_layer" repeated structure.
        # We'll replicate that concept with a small MLP.
        self.middle_layer = middle_layer
        self.batchnorm_mid = batchnorm_mid
        self.dropout_rate_mid = dropout_rate_mid

        # We'll just keep them as placeholders for now; 
        # we'll build them in forward() once we know the flatten size.
        self.middle_dense = None

        # 4) Final: produce z_mean, z_log_var
        self.z_mean_layer = nn.Linear(hidden_dim, hidden_dim)
        self.z_log_var_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_smi, x_cond):
        """
        x_smi: (B, MAX_LEN, NCHARS)
        x_cond: (B, cond_dim)
        """
        batch_size = x_smi.size(0)

        # 1) Move SMILES from (B, MAX_LEN, NCHARS) -> (B, NCHARS, MAX_LEN) for conv1d
        x_smi = x_smi.permute(0, 2, 1)  # shape: (B, input_channels, max_len)

        # 2) Pass through convolution layers
        for i, conv in enumerate(self.conv_layers):
            x_smi = conv(x_smi)  # shape: (B, out_channels, seq_len')
            if self.batchnorm_conv:
                x_smi = self.bn_conv_layers[i](x_smi)
            x_smi = self.activation(x_smi)

        # 3) Flatten
        # shape is (B, out_channels * seq_len')
        x_smi = x_smi.flatten(start_dim=1)

        # 4) Project condition
        cond_out = self.activation(self.cond_proj(x_cond))  # shape (B, cond_dim)

        # 5) Concatenate flattened conv output + condition
        x = torch.cat([x_smi, cond_out], dim=1)

        if self.middle_dense is None:
            in_size = x.size(1)
            layers = []
            hidden_sizes = []
            # For your “middle_layer” concept, replicate that MLP stack:
            # Example: do middle_layer times a Dense with dropout + BN
            current_dim = in_size
            for i in range(self.middle_layer):
                next_dim = self.hidden_dim  # or your growth formula
                dense = nn.Linear(current_dim, next_dim)
                layers.append(dense)
                layers.append(nn.Tanh() if self.activation == F.tanh else nn.ReLU())
                if self.dropout_rate_mid > 0:
                    layers.append(nn.Dropout(self.dropout_rate_mid))
                if self.batchnorm_mid:
                    layers.append(nn.BatchNorm1d(next_dim))
                current_dim = next_dim

            # if no middle_layer, we pass x directly
            # final dimension must be self.hidden_dim
            # we’ll make sure we end with exactly hidden_dim
            if self.middle_layer == 0:
                # Just do a single linear if you want
                layers.append(nn.Linear(in_size, self.hidden_dim))

            self.middle_dense = nn.Sequential(*layers)
            # Move the newly created sub-module to the same device as x
            self.middle_dense = self.middle_dense.to(x.device)
            # Now that it’s built, we freeze it in place.

        x = self.middle_dense(x)

        # 6) Create z_mean, z_log_var
        z_mean = self.z_mean_layer(x)         # shape (B, hidden_dim)
        z_log_var = self.z_log_var_layer(x)   # shape (B, hidden_dim)

        return z_mean, z_log_var


def reparameterize(z_mean, z_log_var):
    """
    z = mu + sigma * epsilon
    """
    std = torch.exp(0.5 * z_log_var)
    eps = torch.randn_like(std)
    return z_mean + eps * std


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        cond_dim,
        n_chars,
        max_len,
        gru_depth=1,
        recurrent_dim=256,
        dropout=0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.max_len = max_len
        self.n_chars = n_chars
        self.gru_depth = gru_depth
        self.recurrent_dim = recurrent_dim

        self.input_proj = nn.Linear(n_chars, recurrent_dim)
        self.latent_to_hidden = nn.Linear(hidden_dim + cond_dim, recurrent_dim)

        # ──► Define ONE GRU with multiple layers
        self.gru = nn.GRU(
            input_size=recurrent_dim,
            hidden_size=recurrent_dim,
            num_layers=gru_depth,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(recurrent_dim, n_chars)

    def forward(self, z, x_cond, teacher_force_inputs=None):
        B = z.size(0)

        # Merge z + condition -> init hidden
        zc = torch.cat([z, x_cond], dim=1)  # (B, hidden_dim + cond_dim)
        h0 = self.latent_to_hidden(zc)     # (B, recurrent_dim)
        
        # For a multi-layer GRU, hidden has shape (num_layers, B, recurrent_dim)
        h0 = h0.unsqueeze(0).repeat(self.gru_depth, 1, 1)

        if teacher_force_inputs is not None:
            # Teacher forcing
            out = self.input_proj(teacher_force_inputs.float())  # (B, seq_len, rdim)
            out, hn = self.gru(out, h0)                          # out: (B, seq_len, rdim)
            out = self.dropout(out)
            logits = self.output_layer(out)                      # (B, seq_len, n_chars)
            return logits

        else:
            # Autoregressive generation
            outputs = []
            input_step = torch.zeros(B, 1, self.recurrent_dim, device=z.device)
            hn = h0
            for t in range(self.max_len):
                out, hn = self.gru(input_step, hn)   # (B, 1, rdim)
                out = self.dropout(out)
                logits = self.output_layer(out)      # (B, 1, n_chars)
                outputs.append(logits)
                # Next input
                input_step = out  # or embed an argmax token, etc.

            outputs = torch.cat(outputs, dim=1)       # (B, max_len, n_chars)
            return outputs


class ConditionalVAE(nn.Module):
    """
    Full CVAE that ties the Encoder + Decoder together.
    """
    def __init__(
        self,
        encoder: ConditionalEncoder,
        decoder: ConditionalDecoder
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_smi, x_cond, teacher_force_inputs=None):
        """
        x_smi: (B, max_len, n_chars)
        x_cond: (B, cond_dim)
        teacher_force_inputs: (B, max_len, n_chars), optional
        """
        # 1) Encoder forward
        z_mean, z_log_var = self.encoder(x_smi, x_cond)

        # 2) Reparameterize
        z = reparameterize(z_mean, z_log_var)

        # 3) Decoder forward
        logits = self.decoder(z, x_cond, teacher_force_inputs=teacher_force_inputs)

        # Return everything you need to compute VAE loss
        return logits, z_mean, z_log_var

    def generate(self, x_cond, max_len=None, device='cuda', teacher_force_inputs=None):
        """
        Generate SMILES given a condition (without teacher forcing).
        """
        if max_len is not None:
            self.decoder.max_len = max_len

        # Usually you sample z ~ N(0,I) at generation time
        # or you can do something else. We’ll do random sampling.
        B = x_cond.size(0)
        z = torch.randn(B, self.encoder.hidden_dim).to(device)

        # Pass to decoder
        outputs = self.decoder(z, x_cond, teacher_force_inputs=teacher_force_inputs)
        # outputs is (B, max_len, n_chars) with logits
        # You can then take argmax along last dim to get actual tokens
        preds = torch.argmax(outputs, dim=-1)  # shape: (B, max_len)
        return preds

