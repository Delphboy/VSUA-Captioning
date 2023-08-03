from __future__ import absolute_import, division, print_function

from argparse import Namespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.utils import expand_feats
from models.Gnns import GNN, GraphAttentionNetwork
from utils.helper import build_embeding_layer, pack_wrapper

from .CaptionModel import CaptionModel


class AttModel(CaptionModel):
    def __init__(self, opt: Namespace) -> None:
        super(AttModel, self).__init__()
        self.opt = opt
        self.geometry_relation = opt.geometry_relation
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.use_bn = getattr(opt, "use_bn", 0)
        self.att_feat_size = opt.att_feat_size
        if opt.use_box:
            self.att_feat_size = self.att_feat_size + 5  # concat box position features
        self.sg_label_embed_size = opt.sg_label_embed_size
        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = build_embeding_layer(
            self.vocab_size + 1, self.input_encoding_size, self.drop_prob_lm
        )
        self.fc_embed = nn.Sequential(
            nn.Linear(self.fc_feat_size, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm),
        )
        self.att_embed = nn.Sequential(
            *(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())
                + (
                    nn.Linear(self.att_feat_size, self.rnn_size),
                    nn.ReLU(),
                    nn.Dropout(self.drop_prob_lm),
                )
                + ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())
            )
        )

        # lazily use the same vocabulary size for obj, attr and rela embeddings
        num_objs = num_attrs = num_relas = 472
        self.obj_embed = build_embeding_layer(
            num_objs, self.sg_label_embed_size, self.drop_prob_lm
        )
        self.attr_embed = build_embeding_layer(
            num_attrs, self.sg_label_embed_size, self.drop_prob_lm
        )
        if not self.geometry_relation:
            self.rela_embed = build_embeding_layer(
                num_relas, self.sg_label_embed_size, self.drop_prob_lm
            )

        self.proj_obj = nn.Sequential(
            *[
                nn.Linear(
                    self.rnn_size
                    + self.sg_label_embed_size * self.opt.num_obj_label_use,
                    self.rnn_size,
                ),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
        )
        self.proj_attr = nn.Sequential(
            *[
                nn.Linear(self.sg_label_embed_size * 3, self.rnn_size),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
        )

        if self.opt.geometry_relation:
            proj_rela_dim = self.opt.geometry_rela_feat_dim
        else:
            proj_rela_dim = self.sg_label_embed_size

        self.proj_rela = nn.Sequential(
            *[
                nn.Linear(
                    proj_rela_dim,
                    self.rnn_size,
                ),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
        )

        # self.gnn = GNN(opt)
        self.gnn = GraphAttentionNetwork(
            in_features=self.rnn_size, out_features=self.rnn_size, n_heads=5, opt=opt
        )

        self.ctx2att_obj = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_attr = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_rela = nn.Linear(self.rnn_size, self.att_hid_size)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embed[0].weight.data.uniform_(-initrange, initrange)
        self.obj_embed[0].weight.data.uniform_(-initrange, initrange)
        self.attr_embed[0].weight.data.uniform_(-initrange, initrange)
        if not self.geometry_relation:
            self.rela_embed[0].weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size: int) -> tuple([torch.Tensor, torch.Tensor]):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.rnn_size),
            weight.new_zeros(self.num_layers, batch_size, self.rnn_size),
        )

    def _embed_vsu(
        self,
        obj_labels: torch.Tensor,
        attr_labels: torch.Tensor,
        rela_labels: torch.Tensor,
    ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        obj_embed = self.obj_embed(obj_labels)
        attr_embed = self.attr_embed(attr_labels)
        if self.geometry_relation:
            rela_embed = rela_labels
        else:
            rela_embed = self.rela_embed(rela_labels)

        return obj_embed, attr_embed, rela_embed

    def _proj_vsu(
        self,
        obj_embed: torch.Tensor,
        attr_embed: torch.Tensor,
        rela_embed: torch.Tensor,
        att_feats: torch.Tensor,
    ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        "project node features, equation 4-7 in paper"

        # handle multiple object labels
        obj_embed = obj_embed.view(obj_embed.size(0), obj_embed.size(1), -1)
        obj_vecs = self.proj_obj(torch.cat([att_feats, obj_embed], dim=-1)) + att_feats

        # handle multiple attribute labels: (128, 3) -> (128*3)
        attr_vecs = attr_embed.view(attr_embed.size(0), attr_embed.size(1), -1)
        attr_vecs = self.proj_attr(attr_vecs)

        rela_vecs = self.proj_rela(rela_embed)
        return obj_vecs, attr_vecs, rela_vecs

    def _prepare_vsu_features(
        self, sg_data: dict, att_feats: torch.Tensor, att_masks: torch.Tensor
    ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        prepare node features for each type of visual semantic units (vsus):
        obj, attr, and rela

        the raw data that are needed:
            - obj_labels: (B, No, 1)
            - attr_labels: (B, No, 3)
            - rela_labels: (B, Nr, ?)
            # - rela_triplets: (subj_index, obj_index, rela_label) of shape (B, Nr, 3)
            - rela_edges: LongTensor of shape (B, Nr, 2),
                        where rela_edges[b, k] = [i, j]
                        indicates the presence of the relation triple:
                        ( obj[b][i], rela[b][k], obj[b][j] ),
                        i.e. the k-th relation of the b-th sample which is between the
                        i-th and j-th objects
        """
        obj_labels = sg_data["obj_labels"]
        attr_labels = sg_data["attr_labels"]
        rela_masks = sg_data["rela_masks"]
        rela_edges = sg_data["rela_edges"]
        rela_labels = sg_data["rela_feats"]
        rela_weights = sg_data.get("rela_weights", None)

        att_masks = att_masks.unsqueeze(-1)
        rela_masks = rela_masks.unsqueeze(-1)

        # node features
        obj_embed, attr_embed, rela_embed = self._embed_vsu(
            obj_labels, attr_labels, rela_labels
        )

        # project node features to the same size as att_feats
        obj_vecs, attr_vecs, rela_vecs = self._proj_vsu(
            obj_embed, attr_embed, rela_embed, att_feats
        )

        # node embedding with simple gnns
        obj_vecs, attr_vecs, rela_embed = self.gnn(
            obj_vecs,
            attr_vecs,
            rela_vecs,  # rela_embed,
            rela_edges,
            rela_masks,
            rela_weights,
        )

        return obj_vecs, attr_vecs, rela_vecs

    def prepare_core_args(
        self,
        sg_data: dict,
        fc_feats: torch.Tensor,
        att_feats: torch.Tensor,
        att_masks: torch.Tensor,
    ) -> list:
        rela_masks = sg_data["rela_masks"]
        # embed fc and att features
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        obj_feats, attr_feats, rela_feats = self._prepare_vsu_features(
            sg_data, att_feats, att_masks
        )

        # Project the attn feats first to reduce memory and computation consumptions
        p_obj_feats = p_attr_feats = p_rela_feats = []
        if "o" in self.opt.vsua_use:
            p_obj_feats = self.ctx2att_obj(obj_feats)
        if "a" in self.opt.vsua_use:
            p_attr_feats = self.ctx2att_attr(attr_feats)
        if "r" in self.opt.vsua_use:
            p_rela_feats = self.ctx2att_rela(rela_feats)

        core_args = [
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
        ]
        return core_args

    def _forward(
        self,
        sg_data: dict,
        fc_feats: torch.Tensor,
        att_feats: torch.Tensor,
        seq: torch.Tensor,
        att_masks=Optional[torch.Tensor],
    ) -> torch.Tensor:
        core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)
        # make seq_per_img copies of the encoded inputs:
        # shape: (B, ...) => (B*seq_per_image, ...)
        core_args = expand_feats(core_args, self.seq_per_img)

        batch_size = fc_feats.size(0) * self.seq_per_img
        state = self.init_hidden(batch_size)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)

        # teacher forcing
        for i in range(seq.size(1) - 1):
            # scheduled sampling
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(
                        outputs[:, i - 1].detach()
                    )  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(
                        0,
                        sample_ind,
                        torch.multinomial(prob_prev, 1)
                        .view(-1)
                        .index_select(0, sample_ind),
                    )
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, state, core_args)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(
        self,
        it: torch.Tensor,
        state: tuple([torch.Tensor, torch.Tensor]),
        core_args: list,
    ) -> tuple([torch.Tensor, tuple([torch.Tensor, torch.Tensor])]):
        # 'it' contains a word index
        xt = self.embed(it)

        # self.core is defined in VSUAModel
        output, state = self.core(xt, state, core_args)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    # sample sentences with greedy decoding
    def _sample(
        self,
        sg_data: dict,
        fc_feats: torch.Tensor,
        att_feats: torch.Tensor,
        att_masks: Optional[torch.Tensor] = None,
        opt: Optional[dict] = {},
        _core_args=None,
    ) -> List[torch.Tensor]:
        sample_max = opt.get("sample_max", 1)
        beam_size = opt.get("beam_size", 1)
        temperature = opt.get("temperature", 1.0)
        decoding_constraint = opt.get("decoding_constraint", 0)
        return_core_args = opt.get("return_core_args", False)
        expand_features = opt.get("expand_features", True)

        if beam_size > 1:
            return self._sample_beam(sg_data, fc_feats, att_feats, att_masks, opt)
        if _core_args is not None:
            # reuse the core_args calculated during generating sampled captions
            # when generating greedy captions for SCST,
            core_args = _core_args
        else:
            core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)

        # make seq_per_img copies of the encoded inputs:
        # shape: (B, ...) => (B*seq_per_image, ...)
        # should be True when training (xe or scst), False when evaluation
        if expand_features:
            if return_core_args:
                _core_args = core_args
            core_args = expand_feats(core_args, self.seq_per_img)
            batch_size = fc_feats.size(0) * self.opt.seq_per_img
        else:
            batch_size = fc_feats.size(0)

        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, state, core_args)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float("-inf"))
                logprobs = logprobs + tmp
            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(
                        logprobs.data
                    )  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(
                    1, it
                )  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        returns = [seq, seqLogprobs]
        if return_core_args:
            returns.append(_core_args)
        return returns

    # sample sentences with beam search
    def _sample_beam(
        self,
        sg_data: dict,
        fc_feats: torch.Tensor,
        att_feats: torch.Tensor,
        att_masks: Optional[torch.Tensor] = None,
        opt: Optional[dict] = {},
    ) -> tuple([torch.Tensor, torch.Tensor]):
        beam_size = opt.get("beam_size", 10)
        batch_size = fc_feats.size(0)

        core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, "lets assume this for now"
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            sample_core_args = []
            for item in core_args:
                if type(item) is list or item is None:
                    sample_core_args.append(item)
                    continue
                else:
                    sample_core_args.append(item[k : k + 1])
            sample_core_args = expand_feats(sample_core_args, beam_size)

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, state, sample_core_args)

            self.done_beams[k] = self.beam_search(
                state, logprobs, sample_core_args, opt=opt
            )
            seq[:, k] = self.done_beams[k][0][
                "seq"
            ]  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]["logps"]
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
