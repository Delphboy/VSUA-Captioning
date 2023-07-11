from __future__ import absolute_import, division, print_function

from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AttnModel import AttModel

"""
The OARs/Rg model proposed in our paper: 
"Aligning Linguistic Words and Visual Semantic Units for Image Captioning"
"""


class Attention(nn.Module):
    def __init__(self, opt: Namespace) -> None:
        super(Attention, self).__init__()
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.query_dim = self.rnn_size
        self.h2att = nn.Linear(self.query_dim, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(
        self,
        h: torch.Tensor,
        att_feats: torch.Tensor,
        p_att_feats: torch.Tensor,
        att_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(
            -1, att_size, att_feats.size(-1)
        )  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(
            1
        )  # batch * att_feat_size
        return att_res


class VSUACore(nn.Module):
    def __init__(self, opt: Namespace, use_maxout: bool = False) -> None:
        super(VSUACore, self).__init__()
        self.opt = opt
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(
            opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size
        )  # we, fc, h^2_t-1

        lang_lstm_in_dim = opt.rnn_size * (1 + len(self.opt.vsua_use))
        self.lang_lstm = nn.LSTMCell(lang_lstm_in_dim, opt.rnn_size)  # h^1_t, \hat v

        if "o" in self.opt.vsua_use:
            self.attention_obj = Attention(opt)
        if "a" in self.opt.vsua_use:
            self.attention_attr = Attention(opt)
        if "r" in self.opt.vsua_use:
            self.attention_rela = Attention(opt)

    def forward(
        self,
        xt: torch.Tensor,
        state: tuple([torch.Tensor, torch.Tensor]),
        core_args: list[torch.Tensor],
    ) -> tuple([torch.Tensor, tuple([torch.Tensor, torch.Tensor])]):
        (
            fc_feats,
            att_feats,
            obj_feats,
            attr_feats,
            rela_feats,
            p_obj_feats,
            p_attr_feats,
            p_rela_feats,
            att_masks,
            rela_masks,
        ) = core_args
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        lang_lstm_input = h_att
        if "o" in self.opt.vsua_use:
            att_obj = self.attention_obj(h_att, obj_feats, p_obj_feats, att_masks)
            lang_lstm_input = torch.cat([lang_lstm_input, att_obj], 1)

        if "a" in self.opt.vsua_use:
            att_attr = self.attention_attr(h_att, attr_feats, p_attr_feats, att_masks)
            lang_lstm_input = torch.cat([lang_lstm_input, att_attr], 1)

        if "r" in self.opt.vsua_use:
            att_rela = self.attention_rela(h_att, rela_feats, p_rela_feats, rela_masks)
            lang_lstm_input = torch.cat([lang_lstm_input, att_rela], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return output, state


class VSUAModel(AttModel):
    def __init__(self, opt: Namespace) -> None:
        super(VSUAModel, self).__init__(opt)
        self.num_layers = 2
        self.core = VSUACore(opt)
        self.core = VSUACore(opt)
        self.core = VSUACore(opt)
