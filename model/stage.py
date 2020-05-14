import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import argparse

from argparse import Namespace
from torch.utils.data import DataLoader
from tvqa_dataset import TVQACommonDataset, TVQASplitDataset, pad_collate, PadCollate
from .context_query_attention import StructuredAttentionWithDownsize
from .encoder import StackedEncoder, StackedEncoderConf
from .cnn import DepthwiseSeparableConv
from .model_utils import save_pickle, mask_logits, flat_list_of_lists, \
    find_max_triples, get_high_iou_sapns, expand_span


class LinearWrapper(nn.Module):
    """1D conv layer"""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearWrapper, self).__init__()
        self.relu = relu
        layers = [nn.LayerNorm(in_hsz)] if layer_norm else []
        layers += [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.relu:
            return F.relu(self.conv(x), inplace=True)  # (N, L, D)
        else:
            return self.conv(x)  # (N, L, D)


class ConvLinear(nn.Module):
    """1D conv layer"""
    def __init__(self, in_hsz, out_hsz, kernel_size=3, layer_norm=True, dropout=0.1, relu=True):
        super(ConvLinear, self).__init__()
        layers = [nn.LayerNorm(in_hsz)] if layer_norm else []
        layers += [
            nn.Dropout(dropout),
            DepthwiseSeparableConv(in_ch=in_hsz,
                                   out_ch=out_hsz,
                                   k=kernel_size,
                                   dim=1,
                                   relu=relu)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        return self.conv(x)  # (N, L, D)


class InputCommonEncoder(nn.Module):
    def __init__(self, bridge_hsz, stacked_enc_conf: StackedEncoderConf):
        super().__init__()

        self.downsize_encoder = nn.Sequential(
            nn.Dropout(stacked_enc_conf.dropout),
            nn.Linear(bridge_hsz, stacked_enc_conf.hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(stacked_enc_conf.hidden_size),
        )

        self.stacked_encoder = StackedEncoder(stacked_enc_conf)

    def forward(self, x, mask):
        x_downsize = self.downsize_encoder(x)
        return self.stacked_encoder(x_downsize, mask)


class InputTextEncoder(nn.Module):
    def __init__(self, wd_size, bridge_hsz, dropout, common_encoder):
        super().__init__()

        self.bert_word_encoding_fc = nn.Sequential(
            nn.LayerNorm(wd_size),
            nn.Dropout(dropout),
            nn.Linear(wd_size, bridge_hsz),
            nn.ReLU(True),
            nn.LayerNorm(bridge_hsz),
        )

        self.common_encoder = common_encoder

    def forward(self, x, mask):
        bert_encoding = self.bert_word_encoding_fc(x)
        return self.common_encoder(bert_encoding, mask)


class InputVideoEncoder(nn.Module):
    def __init__(self, vfeat_size, bridge_hsz, dropout, common_encoder):
        super().__init__()

        self.vid_fc = nn.Sequential(
            nn.LayerNorm(vfeat_size),
            nn.Dropout(dropout),
            nn.Linear(vfeat_size, bridge_hsz),
            nn.ReLU(True),
            nn.LayerNorm(bridge_hsz)
        )

        self.common_encoder = common_encoder

    def forward(self, x, mask):
        vid_encoding = self.vid_fc(x)
        return self.common_encoder(vid_encoding, mask)


class ClassifierHeadMultiProposal(nn.Module):
    def __init__(self, stacked_enc_conf: StackedEncoderConf, hsz, add_local=False, t_iter=0):
        super().__init__()

        self.t_iter = t_iter
        self.add_local = add_local

        self.cls_encoder = StackedEncoder(stacked_enc_conf)

        self.cls_projection_layers = nn.ModuleList(
            [
                LinearWrapper(in_hsz=hsz,
                              out_hsz=hsz,
                              layer_norm=True,
                              dropout=stacked_enc_conf.dropout,
                              relu=True)
            ] +
            [
                ConvLinear(in_hsz=hsz,
                           out_hsz=hsz,
                           kernel_size=3,
                           layer_norm=True,
                           dropout=stacked_enc_conf.dropout,
                           relu=True)
                for _ in range(t_iter)])

        self.temporal_scoring_st_layers = nn.ModuleList([
            LinearWrapper(in_hsz=hsz,
                          out_hsz=1,
                          layer_norm=True,
                          dropout=stacked_enc_conf.dropout,
                          relu=False)
            for _ in range(t_iter+1)])

        self.temporal_scoring_ed_layers = nn.ModuleList([
            LinearWrapper(in_hsz=hsz,
                          out_hsz=1,
                          layer_norm=True,
                          dropout=stacked_enc_conf.dropout,
                          relu=False)
            for _ in range(t_iter+1)])

        self.classifier = LinearWrapper(in_hsz=hsz * 2 if add_local else hsz,
                                        out_hsz=1,
                                        layer_norm=True,
                                        dropout=stacked_enc_conf.dropout,
                                        relu=False)

    def residual_temporal_predictor(self, layer_idx, input_tensor):
        """
        Args:
            layer_idx (int):
            input_tensor: (N, L, D)

        Returns:
            temporal_score
        """
        input_tensor = input_tensor + self.cls_projection_layers[layer_idx](input_tensor)  # (N, L, D)
        t_score_st = self.temporal_scoring_st_layers[layer_idx](input_tensor)  # (N, L, 1)
        t_score_ed = self.temporal_scoring_ed_layers[layer_idx](input_tensor)  # (N, L, 1)
        t_score = torch.cat([t_score_st, t_score_ed], dim=2)  # (N, L, 2)
        return input_tensor, t_score

    def get_proposals(self, max_statement, max_statement_mask, temporal_scores,
                      targets, ts_labels, max_num_proposal=1, iou_thd=0.5, ce_prob_thd=0.01,
                      extra_span_length=3):
        """
        Args:
            max_statement: (N, 5, Li, D)
            max_statement_mask: (N, 5, Li, 1)
            temporal_scores: (N, 5, Li, 2)
            targets: (N, )
            ts_labels: (N, Li) for frm or N * (st, ed) for st_ed
            max_num_proposal:
            iou_thd:
            ce_prob_thd:
            extra_span_length:
        Returns:

        """
        bsz, num_a, num_img, _ = max_statement_mask.shape
        if self.training:
            ca_temporal_scores_st_ed = \
                temporal_scores[torch.arange(bsz, dtype=torch.long), targets].data  # (N, Li, 2)
            ca_temporal_scores_st_ed = F.softmax(ca_temporal_scores_st_ed, dim=1)  # (N, Li, 2)
            ca_pred_spans = find_max_triples(ca_temporal_scores_st_ed[:, :, 0],
                                             ca_temporal_scores_st_ed[:, :, 1],
                                             topN=max_num_proposal,
                                             prob_thd=ce_prob_thd)  # N * [(st_idx, ed_idx, confidence), ...]
            # +1 for ed index before forward into get_high_iou_spans func.
            ca_pred_spans = [[[sub_e[0], sub_e[1] + 1, sub_e[2]] for sub_e in e] for e in ca_pred_spans]
            spans = get_high_iou_sapns(zip(ts_labels["st"].tolist(), (ts_labels["ed"] + 1).tolist()),
                                       ca_pred_spans, iou_thd=iou_thd, add_gt=True)  # N * [(st, ed), ...]
            local_max_max_statement_list = []  # N_new * (5, D)
            global_max_max_statement_list = []  # N_new * (5, D)
            span_targets = []  # N_new * (1,)
            for idx, (t, span_sublist) in enumerate(zip(targets, spans)):
                span_targets.extend([t] * len(span_sublist))
                cur_global_max_max_statement = \
                    torch.max(mask_logits(max_statement[idx], max_statement_mask[idx]), 1)[0]
                global_max_max_statement_list.extend([cur_global_max_max_statement] * len(span_sublist))
                for span in span_sublist:
                    span = expand_span(span, expand_length=extra_span_length)
                    cur_span_max_statement = mask_logits(
                        max_statement[idx, :, span[0]:span[1]],
                        max_statement_mask[idx, :, span[0]:span[1]])  # (5, Li[st:ed], D)
                    local_max_max_statement_list.append(torch.max(cur_span_max_statement, 1)[0])  # (5, D)
            local_max_max_statement = torch.stack(local_max_max_statement_list)  # (N_new, 5, D)
            global_max_max_statement = torch.stack(global_max_max_statement_list)  # (N_new, 5, D)
            max_max_statement = torch.cat([
                local_max_max_statement,
                global_max_max_statement], dim=-1)  # (N_new, 5, 2D)
            return max_max_statement, targets.new_tensor(span_targets)  # (N_new, 5, 2D), (N_new, )
        else:  # testing
            temporal_scores_st_ed = F.softmax(temporal_scores, dim=2)  # (N, 5, Li, 2)
            temporal_scores_st_ed_reshaped = temporal_scores_st_ed.view(bsz * num_a, -1, 2)  # (N*5, Li, 2)
            pred_spans = find_max_triples(temporal_scores_st_ed_reshaped[:, :, 0],
                                          temporal_scores_st_ed_reshaped[:, :, 1],
                                          topN=1, prob_thd=None)  # (N*5) * [(st, ed, confidence), ]
            pred_spans = flat_list_of_lists(pred_spans)  # (N*5) * (st, ed, confidence)
            pred_spans = torch.FloatTensor(pred_spans).to(temporal_scores_st_ed_reshaped.device)  # (N*5, 3)
            pred_spans, pred_scores = pred_spans[:, :2].long(), pred_spans[:, 2]  # (N*5, 2), (N*5, )
            pred_spans = [[e[0], e[1] + 1] for e in pred_spans]
            max_statement = max_statement.view(bsz * num_a, num_img, -1)  # (N*5, Li, D)
            max_statement_mask = max_statement_mask.view(bsz * num_a, num_img, -1)  # (N*5, Li, 1)
            local_max_max_statement_list = []  # N*5 * (D, )
            global_max_max_statement_list = []  # N*5 * (D, )
            for idx, span in enumerate(pred_spans):
                span = expand_span(span, expand_length=extra_span_length)
                cur_global_max_max_statement = \
                    torch.max(mask_logits(max_statement[idx], max_statement_mask[idx]), 0)[0]
                global_max_max_statement_list.append(cur_global_max_max_statement)
                cur_span_max_statement = mask_logits(
                    max_statement[idx, span[0]:span[1]],
                    max_statement_mask[idx, span[0]:span[1]])  # (Li[st:ed], D), words for span[0] == span[1]
                local_max_max_statement_list.append(torch.max(cur_span_max_statement, 0)[0])  # (D, )
            local_max_max_statement = torch.stack(local_max_max_statement_list)  # (N*5, D)
            global_max_max_statement = torch.stack(global_max_max_statement_list)  # (N*5, D)
            max_max_statement = torch.cat([
                local_max_max_statement,
                global_max_max_statement], dim=-1)  # (N_new, 5, 2D)
            return max_max_statement.view(bsz, num_a, -1), targets  # (N, 5, 2D), (N, )

    def forward(self, statement, statement_mask, targets, ts_labels, ts_labels_mask,
                max_num_proposal=1, ce_prob_thd=0.01, iou_thd=0.5, extra_span_length=3):
        """Predict the probabilities of each statements being true. Statements = QA + Context.
        Args:
            statement: (N, 5, Li, Lqa, D)
            statement_mask: (N, 5, Li, Lqa)
            targets: (N, )
            ts_labels: (N, Li) for frm or N * (st, ed) for st_ed
            ts_labels_mask: (N, Li)
            max_num_proposal (int):
            ce_prob_thd (float): threshold for p1*p2 (st, ed)
            iou_thd (float): threshold for temporal iou
            extra_span_length (int): expand the localized span to give a little bit extra context
        Returns:
        """
        bsz, num_a, num_img, num_words = statement_mask.shape
        statement = statement.view(bsz * num_a * num_img, num_words, -1)  # (N*5*Li, Lqa, D)
        statement_mask = statement_mask.view(bsz * num_a * num_img, num_words)  # (N*5*Li, Lqa)
        statement = self.cls_encoder(statement, statement_mask)  # (N*5*Li, Lqa, D)
        max_statement = torch.max(mask_logits(statement, statement_mask.unsqueeze(2)), 1)[0]  # (N*5*Li, D)
        max_statement_mask = (statement_mask.sum(1) != 0).float().view(bsz, num_a, num_img, 1)  # (N, 5, Li, 1)
        max_statement = max_statement.view(bsz * num_a, num_img, -1)  # (N, 5, Li, D)

        t_score_container = []
        encoded_max_statement_container = []
        encoded_max_statement = max_statement  # (N*5, Li, D)
        for layer_idx in range(self.t_iter + 1):
            encoded_max_statement, prev_t_score = \
                self.residual_temporal_predictor(layer_idx, encoded_max_statement)
            t_score_container.append(prev_t_score.view(bsz, num_a, num_img, 2))  # (N, 5, Li, 2)
            encoded_max_statement_container.append(encoded_max_statement)  # (N*5, Li, D)
        if self.t_iter > 0:
            temporal_scores_st_ed = 0.5 * (t_score_container[0] + torch.stack(t_score_container[:1]).mean(0))
        else:
            temporal_scores_st_ed = t_score_container[0]  # (N, 5, Li, 2)

        # mask before softmax
        temporal_scores_st_ed = mask_logits(temporal_scores_st_ed, ts_labels_mask.view(bsz, 1, num_img, 1))

        # when predict answer, only consider 1st level representation !!!
        # since the others are all generated from the 1st level
        stacked_max_statement = encoded_max_statement_container[0].view(bsz, num_a, num_img, -1)  # (N, 5, Li, D)
        if self.add_local:
            max_max_statement, targets = self.get_proposals(
                stacked_max_statement, max_statement_mask, temporal_scores_st_ed,
                targets, ts_labels, max_num_proposal=max_num_proposal, iou_thd=iou_thd,
                ce_prob_thd=ce_prob_thd, extra_span_length=extra_span_length)  # (N, 5, D)
        else:
            max_max_statement = \
                torch.max(mask_logits(stacked_max_statement, max_statement_mask), 2)[0]  # (N, 5, D)
            # targets = targets

        answer_scores = self.classifier(max_max_statement).squeeze(2)  # (N, 5)
        return answer_scores, targets, temporal_scores_st_ed  # (N_new, 5), (N_new, ) (N, 5, Li, 2)


class Stage(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = {}
        for k, v in vars(hparams).items():
            if isinstance(v, list):
                if isinstance(v[0], int):
                    self.hparams[k] = torch.IntTensor(v)
                elif isinstance(v[0], float):
                    self.hparams[k] = torch.FloatTensor(v)
                else:
                    raise ValueError("hparams: only list of floats or ints is supported")
            elif v is None or isinstance(v, (int, float, str, bool, torch.Tensor)):
                self.hparams[k] = v
            else:
                raise ValueError(f"hparams: unsupported type ({k}: {type(v)})")

        self.hparams = Namespace(**self.hparams)

        self.inference_mode = False
        self.sub_flag = hparams.sub_flag
        self.vfeat_flag = hparams.vfeat_flag
        self.use_sup_att = hparams.use_sup_att
        self.num_negatives = hparams.num_negatives
        self.negative_pool_size = hparams.negative_pool_size
        self.num_hard = hparams.num_hard
        self.drop_topk = hparams.drop_topk
        self.extra_span_length = hparams.extra_span_length
        self.att_loss_type = hparams.att_loss_type
        self.margin = hparams.margin
        self.alpha = hparams.alpha
        self.hsz = hparams.hsz
        # self.num_a = 5
        self.concat_ctx = hparams.concat_ctx

        self.wd_size = hparams.embedding_size
        self.use_hard_negatives = False
        self.hard_negative_start = hparams.hard_negative_start

        bridge_hsz = 300

        input_stack_enc_conf = StackedEncoderConf(n_blocks=hparams.input_encoder_n_blocks,
                                                  n_conv=hparams.input_encoder_n_conv,
                                                  kernel_size=hparams.input_encoder_kernel_size,
                                                  num_heads=hparams.input_encoder_n_heads,
                                                  hidden_size=self.hsz,
                                                  dropout=hparams.dropout)

        common_encoder = InputCommonEncoder(bridge_hsz, input_stack_enc_conf)

        self.text_encoder = InputTextEncoder(hparams.embedding_size, bridge_hsz, hparams.dropout, common_encoder)

        if self.vfeat_flag:
            self.vid_encoder = InputVideoEncoder(hparams.vfeat_size, bridge_hsz, hparams.dropout, common_encoder)

        if self.concat_ctx:
            self.concat_fc = nn.Sequential(
                nn.LayerNorm(3 * hparams.hsz),
                nn.Dropout(hparams.dropout),
                nn.Linear(3 * hparams.hsz, hparams.hsz),
                nn.ReLU(True),
                nn.LayerNorm(hparams.hsz),
            )

        self.qa_ctx_attn = StructuredAttentionWithDownsize(
            hparams.hsz,
            dropout=hparams.dropout,
            scale=hparams.scale,
            add_void=hparams.add_non_visual)  # no parameters inside

        cls_stack_enc_conf = StackedEncoderConf(n_blocks=hparams.cls_encoder_n_blocks,
                                                n_conv=hparams.cls_encoder_n_conv,
                                                kernel_size=hparams.cls_encoder_kernel_size,
                                                num_heads=hparams.cls_encoder_n_heads,
                                                hidden_size=self.hsz,
                                                dropout=hparams.dropout)

        self.classfier_head_multi_proposal = ClassifierHeadMultiProposal(cls_stack_enc_conf, hparams.hsz,
                                                                         hparams.add_local, hparams.t_iter)

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.pad_collate = PadCollate(hparams)

        self.scheduler = None

    def forward(self, batch):
        bsz = len(batch["qid"])
        num_a = batch["qas"].shape[1]
        hsz = self.hsz

        # (N*5, L, D)
        a_embed = self.text_encoder(batch["qas_bert"].view(bsz * num_a, -1, self.wd_size),  # (N*5, L, D)
                                    batch["qas_mask"].view(bsz * num_a, -1))                # (N*5, L)
        a_embed = a_embed.view(bsz, num_a, 1, -1, hsz)  # (N, 5, 1, L, D)
        a_mask = batch["qas_mask"].view(bsz, num_a, 1, -1)  # (N, 5, 1, L)

        attended_sub, attended_vid, attended_vid_mask, attended_sub_mask = (None,) * 4
        # other_outputs = {}  # {"pos_noun_mask": batch.qa_noun_masks}  # used to visualization and compute att acc
        if self.sub_flag:
            num_imgs, num_words = batch["sub_bert"].shape[1:3]

            # (N*Li, Lw, D)
            sub_embed = self.text_encoder(batch["sub_bert"].view(bsz * num_imgs, num_words, -1),  # (N*Li, Lw)
                                          batch["sub_mask"].view(bsz * num_imgs, num_words))      # (N*Li, Lw)

            sub_embed = sub_embed.contiguous().view(bsz, 1, num_imgs, num_words, -1)  # (N, Li, Lw, D)
            sub_mask = batch["sub_mask"].view(bsz, 1, num_imgs, num_words)  # (N, 1, Li, Lw)

            attended_sub, attended_sub_mask, sub_raw_s, sub_normalized_s = \
                self.qa_ctx_attn(a_embed, sub_embed, a_mask, sub_mask,
                                 noun_mask=None,
                                 void_vector=None)

            # other_outputs["sub_normalized_s"] = sub_normalized_s
            # other_outputs["sub_raw_s"] = sub_raw_s

        vid_raw_s = None
        if self.vfeat_flag:
            num_imgs, num_regions = batch["vid"].shape[1:3]
            vid_embed = F.normalize(batch["vid"], p=2, dim=-1)  # (N, Li, Lr, D)

            # (N*Li, L, D)
            vid_embed = self.vid_encoder(vid_embed.view(bsz * num_imgs, num_regions, -1),      # (N*Li, Lw)
                                         batch["vid_mask"].view(bsz * num_imgs, num_regions))  # (N*Li, Lr)

            vid_embed = vid_embed.contiguous().view(bsz, 1, num_imgs, num_regions, -1)  # (N, 1, Li, Lr, D)
            vid_mask = batch["vid_mask"].view(bsz, 1, num_imgs, num_regions)  # (N, 1, Li, Lr)

            attended_vid, attended_vid_mask, vid_raw_s, vid_normalized_s = \
                self.qa_ctx_attn(a_embed, vid_embed, a_mask, vid_mask,
                                 noun_mask=None,
                                 void_vector=None)

            # other_outputs["vid_normalized_s"] = vid_normalized_s
            # other_outputs["vid_raw_s"] = vid_raw_s

        if self.concat_ctx:
            concat_input_emb = torch.cat([attended_sub,
                                          attended_vid,
                                          attended_sub * attended_vid], dim=-1)  # (N, 5, Li, Lqa, 3D)
            cls_input_emb = self.concat_fc(concat_input_emb)
            cls_input_mask = attended_vid_mask
        elif self.sub_flag:
            cls_input_emb = attended_sub
            cls_input_mask = attended_sub_mask
        elif self.vfeat_flag:
            cls_input_emb = attended_vid
            cls_input_mask = attended_vid_mask
        else:
            raise NotImplementedError

        out, target, t_scores = self.classfier_head_multi_proposal(
            cls_input_emb, cls_input_mask, batch["target"], batch["ts_label"], batch["ts_label_mask"],
            extra_span_length=self.extra_span_length)

        assert len(out) == len(target)

        # other_outputs["temporal_scores"] = t_scores  # (N, 5, Li) or (N, 5, Li, 2)

        # return out, target, att_predictions, t_scores
        return out, target, t_scores, vid_raw_s

    def on_epoch_start(self):
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.trainer.optimizers[0],
                mode="max",
                factor=0.5,
                patience=10,
                verbose=True
            )

        self.use_hard_negatives = self.current_epoch + 1 > self.hard_negative_start

    def training_step(self, batch, batch_idx):
        try:
            outputs, targets, t_scores, vid_raw_s = self.forward(batch)

            att_loss = 0
            att_predictions = None
            if self.use_sup_att and self.vfeat_flag:
                start_indices = batch["anno_st_idx"]
                try:
                    cur_att_loss, cur_att_predictions = \
                        self.get_att_loss(vid_raw_s, batch["att_labels"], batch["target"],
                                          batch["qas"],
                                          qids=batch["qid"],
                                          q_lens=batch["q_l"],
                                          vid_names=batch["vid_name"],
                                          img_indices=batch["image_indices"],
                                          boxes=batch["boxes"],
                                          start_indices=start_indices,
                                          num_negatives=self.num_negatives,
                                          use_hard_negatives=self.use_hard_negatives,
                                          drop_topk=self.drop_topk)
                except AssertionError as e:
                    print(e)
                    save_pickle(
                        {"batch": batch, "start_indices": start_indices, "vid_raw_s": vid_raw_s},
                        "err_dict.pickle"
                    )
                    import sys
                    sys.exit(1)
                att_loss += cur_att_loss
                att_predictions = cur_att_predictions

            temporal_loss = self.get_ts_loss(temporal_scores=t_scores,
                                             ts_labels=batch["ts_label"],
                                             answer_indices=batch["target"])

            # att_loss = att_loss.sum()
            # temporal_loss = temporal_loss.sum()
            cls_loss = self.criterion(outputs, targets)
            qids = batch["qid"]
            # keep the cls_loss at the same magnitude as only classifying batch_size objects
            cls_loss = cls_loss * (1.0 * len(qids) / len(targets))
            loss = cls_loss + self.hparams.att_weight * att_loss + self.hparams.ts_weight * temporal_loss

            # measure accuracy and record loss
            pred_ids = outputs.argmax(dim=1, keepdim=True)
            targets = batch["target"]
            correct_ids = pred_ids.eq(targets.view_as(pred_ids)).sum().item()

            return {'loss': loss,
                    'cls_loss': cls_loss,
                    'att_loss': att_loss,
                    'temporal_loss': temporal_loss,
                    'train_n_correct': correct_ids,
                    'train_n_ids': len(pred_ids)
                    }
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: ran out of memory, skipping batch")
            else:
                print("RuntimeError {}".format(e))

    def training_epoch_end(self, outputs):
        train_total_loss_mean = 0.0
        train_cls_loss_mean = 0.0
        train_att_loss_mean = 0.0
        train_temporal_loss_mean = 0.0
        for output in outputs:
            train_total_loss = output['loss']
            train_cls_loss = output['cls_loss']
            train_att_loss = output['att_loss']
            train_temporal_loss = output['temporal_loss']

            # reduce manually when using dp
            # if self.use_dp or self.use_ddp2:
            #     train_total_loss = torch.mean(train_total_loss)
            train_total_loss_mean += train_total_loss
            train_cls_loss_mean += train_cls_loss
            train_att_loss_mean += train_att_loss
            train_temporal_loss_mean += train_temporal_loss

        train_total_loss_mean /= len(outputs)
        train_cls_loss_mean /= len(outputs)
        train_att_loss_mean /= len(outputs)
        train_temporal_loss_mean /= len(outputs)

        n_total_correct_ids = sum([out["train_n_correct"] for out in outputs])
        n_total_ids = sum([out["train_n_ids"] for out in outputs])

        accuracy = float(n_total_correct_ids) / float(n_total_ids)

        metric_dict = {'train_loss': train_total_loss_mean, 'train_acc': accuracy}
        logger_logs = {'train_total_loss': train_total_loss_mean,
                       'train_cls_loss': train_cls_loss_mean,
                       'train_att_loss': train_att_loss_mean,
                       'train_temporal_loss': train_temporal_loss_mean,
                       'train_acc': accuracy
                       }

        result = {'progress_bar': metric_dict, 'log': logger_logs}

        return result

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        outputs, _, t_scores, _ = self.forward(batch)

        temporal_loss = self.get_ts_loss(temporal_scores=t_scores,
                                         ts_labels=batch["ts_label"],
                                         answer_indices=batch["target"])

        # temporal_loss = temporal_loss.sum()
        cls_loss = self.criterion(outputs, batch["target"])
        loss = cls_loss + self.hparams.ts_weight * temporal_loss

        # measure accuracy and record loss
        pred_ids = outputs.argmax(dim=1, keepdim=True)
        targets = batch["target"]
        correct_ids = pred_ids.eq(targets.view_as(pred_ids)).sum().item()

        return {'val_loss': loss,
                'valid_cls_loss': cls_loss,
                'valid_temporal_loss': temporal_loss,
                'valid_n_correct': correct_ids,
                'valid_n_ids': len(pred_ids)
                }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        val_total_loss_mean = 0.0
        val_cls_loss_mean = 0.0
        val_temporal_loss_mean = 0.0
        for output in outputs:
            val_total_loss = output['val_loss']
            val_cls_loss = output['valid_cls_loss']
            val_temporal_loss = output['valid_temporal_loss']

            # # reduce manually when using dp
            # if self.use_dp or self.use_ddp2:
            #     val_loss = torch.mean(val_loss)

            val_total_loss_mean += val_total_loss
            val_cls_loss_mean += val_cls_loss
            val_temporal_loss_mean += val_temporal_loss

        val_total_loss_mean /= len(outputs)
        val_cls_loss_mean /= len(outputs)
        val_temporal_loss_mean /= len(outputs)

        n_total_correct_ids = sum([out["valid_n_correct"] for out in outputs])
        n_total_ids = sum([out["valid_n_ids"] for out in outputs])

        accuracy = float(n_total_correct_ids) / float(n_total_ids)

        if self.scheduler is not None:
            self.scheduler.step(accuracy)

        metric_dict = {'val_loss': val_total_loss_mean, 'val_acc': accuracy}
        logger_logs = {'valid_total_loss': val_total_loss_mean,
                       'valid_cls_loss': val_cls_loss_mean,
                       'valid_temporal_loss': val_temporal_loss_mean,
                       'valid_acc': accuracy
                       }

        result = {'progress_bar': metric_dict, 'log': logger_logs}

        return result

    # def test_step(self, batch, batch_idx):
    #     # OPTIONAL
    #     pass

    # def test_end(self, outputs):
    #     # OPTIONAL
    #     pass

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd)

        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.5,
        #     patience=10,
        #     verbose=True
        # )
        #
        # return [optimizer], [scheduler]

    def train_dataloader(self):
        common_dset = TVQACommonDataset(self.hparams)
        train_dset = TVQASplitDataset(common_dset, self.hparams.train_path, "train")
        # self.hparams.vocab_size = len(common_dset.word2idx)

        train_loader = DataLoader(train_dset, batch_size=self.hparams.bsz, shuffle=True,
                                  collate_fn=self.pad_collate,
                                  num_workers=self.hparams.num_workers, pin_memory=True)

        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        common_dset = TVQACommonDataset(self.hparams)
        valid_dset = TVQASplitDataset(common_dset, self.hparams.valid_path, "valid")

        valid_loader = DataLoader(valid_dset, batch_size=self.hparams.test_bsz, shuffle=False,
                                  collate_fn=self.pad_collate, num_workers=self.hparams.num_workers, pin_memory=True)

        return valid_loader

    # def test_dataloader(self):
    #     # OPTIONAL
    #     # can also return a list of test dataloaders
    #     pass

    def get_ts_loss(self, temporal_scores, ts_labels,  answer_indices):
        """
        Args:
            temporal_scores: (N, 5, Li, 2)
            ts_labels: dict(st=(N, ), ed=(N, ))
            answer_indices: (N, )

        Returns:

        """
        bsz = len(answer_indices)
        # compute loss
        ca_temporal_scores_st_ed = \
            temporal_scores[torch.arange(bsz, dtype=torch.long), answer_indices]  # (N, Li, 2)
        loss_st = self.criterion(ca_temporal_scores_st_ed[:, :, 0], ts_labels["st"])
        loss_ed = self.criterion(ca_temporal_scores_st_ed[:, :, 1], ts_labels["ed"])

        return (loss_st + loss_ed) / 2.

    @classmethod
    def sample_negatives(cls, pred_score, pos_indices, neg_indices, num_negatives=2,
                         use_hard_negatives=False, negative_pool_size=0, num_hard=2, drop_topk=0):
        """ Sample negatives from a set of indices. Several sampling strategies are supported:
        1, random; 2, hard negatives; 3, drop_topk hard negatives; 4, mix easy and hard negatives
        5, sampling within a pool of hard negatives; 6, sample across images of the same video.
        Args:
            pred_score: (num_img, num_words, num_region)
            pos_indices: (N_pos, 3) all positive region indices for the same word, not necessaryily the same image.
            neg_indices: (N_neg, 3) ...
            num_negatives (int):
            use_hard_negatives (bool):
            negative_pool_size (int):
            num_hard (int):
            drop_topk (int):
        Returns:

        """
        num_unique_pos = len(pos_indices)
        sampled_pos_indices = torch.cat([pos_indices] * num_negatives, dim=0)
        if use_hard_negatives:
            # print("using use_hard_negatives")
            neg_scores = pred_score[neg_indices[:, 0], neg_indices[:, 1], neg_indices[:, 2]]  # TODO
            max_indices = torch.sort(neg_scores, descending=True)[1].tolist()
            if negative_pool_size > num_negatives:  # sample from a pool of hard negatives
                hard_pool = max_indices[drop_topk:drop_topk + negative_pool_size]
                hard_pool_indices = neg_indices[hard_pool]
                num_hard_negs = num_negatives
                sampled_easy_neg_indices = []
                if num_hard < num_negatives:
                    easy_pool = max_indices[drop_topk + negative_pool_size:]
                    easy_pool_indices = neg_indices[easy_pool]
                    num_hard_negs = num_hard
                    num_easy_negs = num_negatives - num_hard_negs
                    sampled_easy_neg_indices = easy_pool_indices[
                        torch.randint(low=0, high=len(easy_pool_indices),
                                      size=(num_easy_negs * num_unique_pos, ), dtype=torch.long)
                    ]
                sampled_hard_neg_indices = hard_pool_indices[
                    torch.randint(low=0, high=len(hard_pool_indices),
                                  size=(num_hard_negs * num_unique_pos, ), dtype=torch.long)
                ]

                if len(sampled_easy_neg_indices) != 0:
                    sampled_neg_indices = torch.cat([sampled_hard_neg_indices, sampled_easy_neg_indices], dim=0)
                else:
                    sampled_neg_indices = sampled_hard_neg_indices

            else:  # directly take the top negatives
                sampled_neg_indices = neg_indices[max_indices[drop_topk:drop_topk+len(sampled_pos_indices)]]
        else:
            sampled_neg_indices = neg_indices[
                torch.randint(low=0, high=len(neg_indices), size=(len(sampled_pos_indices),), dtype=torch.long)
            ]
        return sampled_pos_indices, sampled_neg_indices

    def get_att_loss(self, scores, att_labels, target, words, vid_names, qids, q_lens, img_indices, boxes,
                     start_indices, num_negatives=2, use_hard_negatives=False, drop_topk=0):
        """ compute ranking loss, use for loop to find the indices,
        use advanced indexing to perform the real calculation
        Build a list contains a quaduple

        Args:
            scores: cosine similarity scores (N, 5, Li, Lqa, Lr), in the range [-1, 1]
            att_labels: list(tensor), each has dimension (#num_imgs, #num_words, #regions), not batched
            target: 1D tensor (N, )
            words: LongTensor (N, 5, Lqa)
            vid_names: list(str) (N,)
            qids: list(int), (N, )
            q_lens: list(int), (N, )
            img_indices: list(list(int)), (N, Li), or None
            boxes: list(list(box)) of length N, each sublist represent an image,
                each box contains the coordinates of xyxy, or None
            num_negatives: number of negatives for each positive region
            use_hard_negatives: use hard negatives, uselect negatives with high scores
            drop_topk: drop topk highest negatives (since the top negatives might be correct, they are just not labeled)
            start_indices (list of int): each element is an index (at 0.5fps) of the first image
                with spatial annotation. If with_ts, set to zero
        Returns:
            att_loss: loss value for the batch
            att_predictions: (list) [{"gt": gt_scores, "pred": pred_scores}, ], used to calculate att. accuracy
        """
        pos_container = []  # contains tuples of 5 elements, which are (batch_i, ca_i, img_i, word_i, region_i)
        neg_container = []
        for batch_idx in range(len(target)):  # batch
            ca_idx = target[batch_idx].cpu().item()
            gt_score = att_labels[batch_idx]  # num_img * (num_words, num_region)
            start_idx = start_indices[batch_idx]  # int
            num_img = len(gt_score)
            sen_l, _ = gt_score[0].shape
            pred_score = scores[batch_idx, ca_idx, :num_img, :sen_l]  # (num_img, num_words, num_region)

            # find positive and negative indices
            batch_pos_indices = []
            batch_neg_indices = []
            for img_idx, img_gt_score in enumerate(gt_score):
                img_idx = start_idx + img_idx
                img_pos_indices = torch.nonzero(img_gt_score)  # (N_pos, 2) ==> (#words, #regions)
                if len(img_pos_indices) == 0:  # skip if no positive indices
                    continue
                img_pos_indices = torch.cat([img_pos_indices.new_full([len(img_pos_indices), 1], img_idx),
                                             img_pos_indices], dim=1)  # (N_pos, 3) ==> (#img, #words, #regions)

                img_neg_indices = torch.nonzero(img_gt_score == 0)  # (N_neg, 2)
                img_neg_indices = torch.cat([img_neg_indices.new_full([len(img_neg_indices), 1], img_idx),
                                             img_neg_indices], dim=1)  # (N_neg, 3)

                batch_pos_indices.append(img_pos_indices)
                batch_neg_indices.append(img_neg_indices)

            if len(batch_pos_indices) == 0:  # skip if empty ==> no gt label for the video
                continue
            batch_pos_indices = torch.cat(batch_pos_indices, dim=0)  # (N_pos, 3) -->
            batch_neg_indices = torch.cat(batch_neg_indices, dim=0)  # (N_neg, 3)

            # sample positives and negatives
            available_img_indices = batch_pos_indices[:, 0].unique().tolist()
            for img_idx in available_img_indices:
                # pos_indices for a certrain img
                img_idx_pos_indices = batch_pos_indices[batch_pos_indices[:, 0] == img_idx]
                img_idx_neg_indices = batch_neg_indices[batch_neg_indices[:, 0] == img_idx]
                available_word_indices = img_idx_pos_indices[:, 1].unique().tolist()
                for word_idx in available_word_indices:
                    # positives and negatives for a given image-word pair, specified by img_idx-word_idx
                    img_idx_word_idx_pos_indices = img_idx_pos_indices[img_idx_pos_indices[:, 1] == word_idx]
                    img_idx_word_idx_neg_indices = img_idx_neg_indices[img_idx_neg_indices[:, 1] == word_idx]
                    # actually all the positives, not sampled pos
                    sampled_pos_indices, sampled_neg_indices = \
                        self.sample_negatives(pred_score,
                                              img_idx_word_idx_pos_indices, img_idx_word_idx_neg_indices,
                                              num_negatives=num_negatives, use_hard_negatives=use_hard_negatives,
                                              negative_pool_size=self.negative_pool_size,
                                              num_hard=self.num_hard, drop_topk=drop_topk)

                    base_indices = torch.LongTensor([[batch_idx, ca_idx]] * len(sampled_pos_indices)).\
                        to(sampled_pos_indices.device)
                    pos_container.append(torch.cat([base_indices, sampled_pos_indices], dim=1))
                    neg_container.append(torch.cat([base_indices, sampled_neg_indices], dim=1))

        pos_container = torch.cat(pos_container, dim=0)
        neg_container = torch.cat(neg_container, dim=0)

        # contain all the predictions and gt labels in this batch, only consider the ones with gt labels
        # also only consider the positive answer.
        att_predictions = None
        if not self.training and self.vfeat_flag:
            att_predictions = dict(det_q=[],
                                   det_ca=[])
            unique_pos_container = np.unique(pos_container.cpu().numpy(), axis=0)  # unique rows in the array
            for row in unique_pos_container:
                batch_idx, ca_idx, img_idx, word_idx, region_idx = row
                start_idx = start_indices[batch_idx]  # int
                cur_q_len = q_lens[batch_idx]
                num_region = att_labels[batch_idx][img_idx-start_idx].shape[1]  # num_img * (num_words, num_region)
                if len(scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu()) != \
                        len(boxes[batch_idx][img_idx-start_idx]):
                    print("scores[batch_idx, ca_idx, img_idx, word_idx].data.cpu()",
                          len(scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu()))
                    print("len(boxes[batch_idx][img_idx-start_idx])", len(boxes[batch_idx][img_idx-start_idx]))
                    print("boxes, batch_idx, img_idx, start_idx, img_idx - start_idx, word_idx",
                          batch_idx, img_idx, start_idx, img_idx - start_idx, word_idx)
                    print(row)
                    raise AssertionError
                cur_det_data = {
                        "pred": scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu(),
                        "word": words[batch_idx, ca_idx, word_idx],
                        "qid": qids[batch_idx],
                        "vid_name": vid_names[batch_idx],
                        "img_idx": img_indices[batch_idx][img_idx],  # full indices
                        "boxes": boxes[batch_idx][img_idx-start_idx]  # located boxes
                    }
                if word_idx < cur_q_len:
                    att_predictions["det_q"].append(cur_det_data)
                else:
                    att_predictions["det_ca"].append(cur_det_data)

        pos_scores = scores[pos_container[:, 0], pos_container[:, 1], pos_container[:, 2],
                            pos_container[:, 3], pos_container[:, 4]]
        neg_scores = scores[neg_container[:, 0], neg_container[:, 1], neg_container[:, 2],
                            neg_container[:, 3], neg_container[:, 4]]

        if self.att_loss_type == "hinge":
            # max(0, m + S_pos - S_neg)
            att_loss = torch.clamp(self.margin + neg_scores - pos_scores, min=0).sum()
        elif self.att_loss_type == "lse":
            # log[1 + exp(scale * (S_pos - S_neg))]
            att_loss = torch.log1p(torch.exp(self.alpha * (neg_scores - pos_scores))).sum()
        else:
            raise NotImplementedError("Only support hinge and lse")
        return att_loss, att_predictions

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        model_arg_parser = parser.add_argument_group("model", description="parser for model arguments")
        model_arg_parser.add_argument("--t_iter", type=int, default=0,
                                      help="positive integer, indicating #iterations for refine temporal prediction")
        model_arg_parser.add_argument("--extra_span_length", type=int, default=3,
                                      help="expand the boundary of the localized span, "
                                           "by [max(0, pred_st - extra_span_length), pred_ed + extra_span_length]")
        model_arg_parser.add_argument("--ts_weight", type=float, default=0.5, help="temporal loss weight")
        model_arg_parser.add_argument("--add_local", action="store_true",
                                      help="concat local feature with global feature for QA")
        # parser.add_argument("--clip", type=float, default=10., help="perform gradient clip")
        model_arg_parser.add_argument("--add_non_visual", action="store_true",
                                      help="count non_visual vectors in when doing weighted sum"
                                           " of the regional vectors")
        model_arg_parser.add_argument("--use_sup_att", action="store_true",
                                      help="supervised att, used with use_noun_mask")
        model_arg_parser.add_argument("--att_weight", type=float, default=0.1, help="weight to att loss")
        model_arg_parser.add_argument("--att_iou_thd", type=float, default=0.5, help="IoU threshold for att label")
        model_arg_parser.add_argument("--margin", type=float, default=0.1, help="margin for ranking loss")
        model_arg_parser.add_argument("--num_region", type=int, default=25, help="max number of regions for each image")
        model_arg_parser.add_argument("--att_loss_type", type=str, default="lse", choices=["hinge", "lse"],
                                      help="att loss type, can be hinge loss or its smooth approximation LogSumExp")
        model_arg_parser.add_argument("--scale", type=float, default=10.,
                                      help="multiplier to be applied to similarity score")
        model_arg_parser.add_argument("--alpha", type=float, default=20.,
                                      help="log1p(1 + exp(m + alpha * x)), "
                                           "a high value penalize more when x > 0, less otherwise")
        model_arg_parser.add_argument("--num_hard", type=int, default=2,
                                      help="number of hard negatives, num_hard<=num_negatives")
        model_arg_parser.add_argument("--num_negatives", type=int, default=2,
                                      help="max number of negatives in ranking loss")
        model_arg_parser.add_argument("--hard_negative_start", type=int, default=100,
                                      help="use hard negative when num epochs > hard_negative_start, "
                                           "set to a very high number to stop using it, e.g. 100")
        model_arg_parser.add_argument("--negative_pool_size", type=int, default=0,
                                      help="sample from a pool of hard negative samples (with high scores), "
                                           "instead of a topk hard ones. "
                                           "directly sample topk when negative_pool_size <= num_negatives")
        model_arg_parser.add_argument("--drop_topk", type=int, default=0,
                                      help="do not use the topk negatives")
        model_arg_parser.add_argument("--embedding_size", type=int, default=768, help="word embedding dim")
        model_arg_parser.add_argument("--hsz", type=int, default=128, help="hidden size.")
        model_arg_parser.add_argument("--vfeat_size", type=int, default=300, help="dimension of the video feature")
        model_arg_parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
        model_arg_parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        model_arg_parser.add_argument("--input_encoder_n_blocks", type=int, default=1)
        model_arg_parser.add_argument("--input_encoder_n_conv", type=int, default=2)
        model_arg_parser.add_argument("--input_encoder_kernel_size", type=int, default=7)
        model_arg_parser.add_argument("--input_encoder_n_heads", type=int, default=0,
                                      help="number of self-attention heads, 0: do not use it")
        model_arg_parser.add_argument("--cls_encoder_n_blocks", type=int, default=1)
        model_arg_parser.add_argument("--cls_encoder_n_conv", type=int, default=2)
        model_arg_parser.add_argument("--cls_encoder_kernel_size", type=int, default=5)
        model_arg_parser.add_argument("--cls_encoder_n_heads", type=int, default=0,
                                      help="number of self-attention heads, 0: do not use it")

        data_arg_parser = parser.add_argument_group("data", description="parser for data arguments")
        data_arg_parser.add_argument("--max_sub_l", type=int, default=50,
                                     help="maxmimum length of all sub sentence 97.71 under 50 for 3 sentences")
        data_arg_parser.add_argument("--max_vid_l", type=int, default=300,
                                     help="maxmimum length of all video sequence")
        data_arg_parser.add_argument("--max_vcpt_l", type=int, default=300,
                                     help="maxmimum length of video seq, 94.25% under 20")
        data_arg_parser.add_argument("--max_q_l", type=int, default=20,
                                     help="maxmimum length of question, 93.91% under 20")  # 25
        data_arg_parser.add_argument("--max_a_l", type=int, default=15,
                                     help="maxmimum length of answer, 98.20% under 15")
        data_arg_parser.add_argument("--max_qa_l", type=int, default=40,
                                     help="maxmimum length of answer, 99.7% <= 40")
        data_arg_parser.add_argument("--word2idx_path", type=str)
        data_arg_parser.add_argument("--qa_bert_path", type=str, default="")
        data_arg_parser.add_argument("--sub_bert_path", type=str, default="")
        data_arg_parser.add_argument("--vcpt_path", type=str, default="")
        data_arg_parser.add_argument("--vfeat_path", type=str, default="")
        data_arg_parser.add_argument("--sub_path", type=str, default="")
        data_arg_parser.add_argument("--frm_cnt_path", type=str, default="")
        data_arg_parser.add_argument("--no_core_driver", action="store_true",
                                     help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        data_arg_parser.add_argument("--h5driver", type=str, default="None", choices=["None", "core"],
                                     help="HDF5 driver. two options are supported: "
                                          "`None` (default) - Use the standard HDF5 driver appropriate. "
                                          "`core` - load into RAM")

        train_arg_parser = parser.add_argument_group("train", description="parser for training arguments")
        train_arg_parser.add_argument("--gradient_clip_val", type=float, default=10., help="perform gradient clip")
        train_arg_parser.add_argument("--train_path", type=str)
        train_arg_parser.add_argument("--valid_path", type=str)

        test_arg_parser = parser.add_argument_group("test", description="parser for test arguments")
        # test_arg_parser.add_argument("--eval_object_vocab_path", type=str)
        test_arg_parser.add_argument("--test_path", type=str, default="")

        return parser

    @staticmethod
    def verify_hparams(hparams):
        v_feat_valid = not (('vfeat_path' in hparams) ^ ('vcpt_path' in hparams))
        if not v_feat_valid:
            msg = f"'vfeat_path' and 'vcpt_path' should be both valid or empty"
            raise ValueError(msg)
        hparams.vfeat_flag = 'vfeat_path' in hparams

        sub_feat_valid = not (('sub_path' in hparams) ^ ('sub_bert_path' in hparams))
        if not sub_feat_valid:
            msg = f"'sub_path' and 'sub_bert_path' should be both valid or empty"
            raise ValueError(msg)
        hparams.sub_flag = 'sub_path' in hparams

        hparams.concat_ctx = hparams.vfeat_flag and hparams.sub_flag

        return hparams
