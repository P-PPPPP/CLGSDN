import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embed import DataEmbedding, DataEmbedding_wo_pos
from .AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class AUTOFORMER_Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(AUTOFORMER_Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.label_len = configs['label_len']
        self.pred_len = configs['pred_len']
        self.output_attention = configs['output_attention']

        # Decomp
        kernel_size = configs['moving_avg']
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs['enc_in'], configs['d_model'], configs['embed'], configs['freq'],
                                                  configs['dropout'])
        self.dec_embedding = DataEmbedding_wo_pos(configs['dec_in'], configs['d_model'], configs['embed'], configs['freq'],
                                                  configs['dropout'])

        # Encoder可以存attn_layers,conv_layers,norm_layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs['factor'], attention_dropout=configs['dropout'],
                                        output_attention=configs['output_attention']),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=my_Layernorm(configs['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs['factor'], attention_dropout=configs['dropout'],
                                        output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs['factor'], attention_dropout=configs['dropout'],
                                        output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['c_out'],
                    configs['d_ff'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                    activation=configs['activation'],
                )
                for l in range(configs['d_layers'])
            ],
            norm_layer=my_Layernorm(configs['d_model']),
            projection=nn.Linear(configs['d_model'], configs['c_out'], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init(32,96,7),(32,96,5);(32,144,7),(32,144,5) #(trend-->mean,season-->zeros)
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)#利用x_enc,pred_len计算mean (batch_size,pred_len,channels)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)#利用x_dec,pred_len计算zeros (batch_size,pred_len,channels)
        seasonal_init, trend_init = self.decomp(x_enc) #(batch_size,inp_len,channels)
        # decoder input( 将label_len与pred_len拼接在一起 (seasonal和trend) )
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1) #(batch_size,pred_len+label_len,channels)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1) #(batch_size,pred_len+label_len,channels)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc) #利用x_enc,x_mark_enc得到enc_out的结果 
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)#enc_out:(batchsize,inp_len,d_model)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec) #(batchsize,inp_len+label_len,channels)
        #(dec_out:(batchsize,pred_len+label_len,d_model),
        # enc_out:(batchsize,inp_len,d_model))
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
