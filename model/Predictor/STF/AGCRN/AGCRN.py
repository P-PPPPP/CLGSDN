import torch
import torch.nn as nn
from .AGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, configs, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1, with_GraphGen=False):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(configs, node_num, dim_in, dim_out, cheb_k, embed_dim, with_GraphGen))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(configs, node_num, dim_out, dim_out, cheb_k, embed_dim, with_GraphGen))

    def forward(self, x, init_state, node_embeddings, adjs=None):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, adjs)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, configs, with_GraphGen=False):
        super(AGCRN, self).__init__()
        self.with_GraphGen = with_GraphGen
        self.num_node = configs['num_nodes']
        self.horizon = configs['seq_len']
        n_nodes = configs['num_nodes']

        self.input_dim = configs['data_channels']
        self.rnn_units = configs['rnn_units']
        self.embedding_dim = configs['embedding_dim']
        self.output_dim = configs['data_channels']
        self.num_layers = configs['num_layers']
        self.cheb_k = configs['cheb_order']

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embedding_dim, requires_grad=True))
        self.encoder = AVWDCRNN(configs, n_nodes, self.input_dim, self.rnn_units, self.cheb_k,
                                self.embedding_dim, self.num_layers, with_GraphGen)

        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.rnn_units), bias=True)

    def forward(self, source, adjs=None):
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, adjs)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 2, 3, 1)                             #B, T, N, C
        return output