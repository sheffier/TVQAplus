from __future__ import annotations
import h5py
import numpy as np
import torch
import copy
from typing import Optional
from torch.utils.data.dataset import Dataset
from pudb.remote import set_trace

from utils import load_pickle, load_json, files_exist, get_all_img_ids, computeIoU, \
    flat_list_of_lists, match_stanford_tokenizer, get_elements_variable_length, dissect_by_lengths


def filter_list_dicts(list_dicts, key, values):
    """ filter out the dicts with values for key"""
    return [e for e in list_dicts if e[key] in values]


def rm_empty_by_copy(list_array):
    """copy the last non-empty element to replace the empty ones"""
    for idx in range(len(list_array)):
        if len(list_array[idx]) == 0:
            list_array[idx] = list_array[idx-1]
    return list_array


class TVQASplitDataset(Dataset):
    def __init__(self, common_dset, split_path, split_mode, debug=False):
        self.common_dset = common_dset
        self.raw_dset = load_json(split_path)
        self.split_mode = split_mode

        if debug:
            self.raw_dset = filter_list_dicts(self.raw_dset, "vid_name", self.common_dset.vcpt_dict.keys())

        self.dset_len = len(self.raw_dset)
        print(f"[{self.split_mode} dataset] {self.dset_len} samples")

    def __len__(self):
        return self.dset_len

    def __getitem__(self, index):
        try:
            raw_dset_sample = self.raw_dset[index]
        except:
            breakpoint()

        try:
            items = self.common_dset[(raw_dset_sample, self.split_mode)]
        except:
            breakpoint()

        return items


class SingletonMeta(type):
    _instance: Optional[TVQACommonDataset] = None

    def __call__(self, args) -> TVQACommonDataset:
        if self._instance is None:
            self._instance = super().__call__(args)
        return self._instance


class TVQACommonDataset(metaclass=SingletonMeta):
    def __init__(self, hparams):
        if hparams.h5driver == "None":
            hparams.h5driver = None
        self.hparams = hparams
        self.sub_data = load_json(hparams.sub_path)
        self.sub_flag = hparams.sub_flag
        self.vfeat_flag = hparams.vfeat_flag
        self.qa_bert_h5 = None
        if self.sub_flag:
            self.sub_bert_h5 = None
        if self.vfeat_flag:
            self.vid_h5 = None
            self.vcpt_dict = load_pickle(hparams.vcpt_path) if hparams.vcpt_path.endswith(".pickle") \
                else load_json(hparams.vcpt_path)

        self.use_sup_att = hparams.use_sup_att
        self.att_iou_thd = hparams.att_iou_thd

        # tmp
        self.frm_cnt_path = hparams.frm_cnt_path
        self.frm_cnt_dict = load_json(self.frm_cnt_path)

        # build/load vocabulary
        assert files_exist([hparams.word2idx_path]), "\nNo cache founded."

        print("\nLoading cache ...")
        self.word2idx = load_json(hparams.word2idx_path)
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.max_num_regions = hparams.num_region

    def __getitem__(self, key):
        sample, mode = key
        if self.qa_bert_h5 is None:
            self.qa_bert_h5 = h5py.File(self.hparams.qa_bert_path, "r", driver=self.hparams.h5driver)  # qid + key
        if self.sub_flag and self.sub_bert_h5 is None:
            self.sub_bert_h5 = h5py.File(self.hparams.sub_bert_path, "r", driver=self.hparams.h5driver)  # vid_name
        if self.vfeat_flag and self.vid_h5 is None:
            self.vid_h5 = h5py.File(self.hparams.vfeat_path, "r", driver=self.hparams.h5driver)  # add core

        # 0.5 fps mode
        items = dict()
        items["vid_name"] = sample["vid_name"]
        vid_name = items["vid_name"]
        items["qid"] = sample["qid"]
        qid = items["qid"]  # int
        frm_cnt = self.frm_cnt_dict[vid_name]
        located_img_ids = sorted([int(e) for e in sample["bbox"].keys()])
        start_img_id, end_img_id = located_img_ids[0], located_img_ids[-1]
        indices, start_idx, end_idx = get_all_img_ids(start_img_id, end_img_id, frm_cnt, frame_interval=6)
        items["anno_st_idx"] = start_idx
        indices = np.array(indices) - 1  # since the frame (image) index from 1

        if "ts" in sample:
            items["ts_label"] = self.get_ts_label(sample["ts"][0],
                                                  sample["ts"][1],
                                                  frm_cnt,
                                                  indices,
                                                  fps=3)
            items["ts"] = sample["ts"]  # [st (float), ed (float)]
        else:
            items["ts_label"], items["ts"] = [0, 0], None
        items["image_indices"] = (indices + 1).tolist()
        items["image_indices"] = items["image_indices"]

        # add q-answers
        answer_keys = ["a0", "a1", "a2", "a3", "a4"]
        qa_sentences = [self.numericalize(sample["q"]
                        + " " + sample[k], eos=False) for k in answer_keys]
        qa_sentences_bert = [
            torch.from_numpy(np.concatenate(
                [self.qa_bert_h5[str(qid) + "_q"], self.qa_bert_h5[str(qid) + "_" + k]],
                axis=0))
            for k in answer_keys]
        q_l = sample["q_len"]
        items["q_l"] = q_l
        items["qas"] = qa_sentences
        items["qas_bert"] = qa_sentences_bert

        # add sub
        if self.sub_flag:
            img_aligned_sub_indices, raw_sub_n_tokens = self.get_aligned_sub_indices(
                indices + 1,
                self.sub_data[vid_name]["sub_text"],
                self.sub_data[vid_name]["sub_time"],
                mode="nearest")
            try:
                sub_bert_embed = dissect_by_lengths(self.sub_bert_h5[vid_name][:], raw_sub_n_tokens, dim=0)
            except AssertionError as e:  # 35 QAs from 7 videos
                sub_bert_embed = dissect_by_lengths(self.sub_bert_h5[vid_name][:], raw_sub_n_tokens,
                                                    dim=0, assert_equal=False)
                sub_bert_embed = rm_empty_by_copy(sub_bert_embed)
            assert len(sub_bert_embed) == len(raw_sub_n_tokens)  # we did not truncate when extract embeddings

            items["sub_bert"] = \
                [torch.from_numpy(
                    np.concatenate(
                        [sub_bert_embed[in_idx] for in_idx in e], axis=0))
                 for e in img_aligned_sub_indices]

            aligned_sub_text = self.get_aligned_sub(self.sub_data[vid_name]["sub_text"],
                                                    img_aligned_sub_indices)
            items["sub"] = [self.numericalize(e, eos=False)
                            for e in aligned_sub_text]
        else:
            items["sub_bert"] = [torch.zeros(2, 2)] * 2
            items["sub"] = [torch.zeros(2, 2)] * 2

        if self.vfeat_flag:
            region_counts = self.vcpt_dict[vid_name]["counts"]  # full resolution

            lowered_vfeat = get_elements_variable_length(
                self.vid_h5[vid_name][:], indices, cnt_list=region_counts, max_num_region=self.max_num_regions)
            cur_vfeat = lowered_vfeat

            items["vfeat"] = [torch.from_numpy(e) for e in cur_vfeat]
        else:
            region_counts = None
            items["vfeat"] = [torch.zeros(2, 2)] * 2

        # add att
        inference = mode == "test"
        if "answer_idx" in sample:
            # add correct answer
            ca_idx = int(sample["answer_idx"])
            items["target"] = ca_idx
            ca_l = sample["a{}_len".format(ca_idx)]

            if self.use_sup_att and not inference and self.vfeat_flag:
                q_ca_sentence = sample["q"] + " " + \
                                sample["a{}".format(ca_idx)]
                iou_data = self.get_iou_data(sample["bbox"], self.vcpt_dict[vid_name], frm_cnt)
                localized_lowered_region_counts = \
                    [min(region_counts[idx], self.max_num_regions) for idx in indices][start_idx:end_idx + 1]
                items["att_labels"] = self.mk_att_label(
                    iou_data, q_ca_sentence, localized_lowered_region_counts, q_l + ca_l + 1,
                    iou_thd=self.att_iou_thd, single_box=inference)
            else:
                items["att_labels"] = None
        else:
            items["target"] = 999  # fake

        return items

    @classmethod
    def get_ts_label(cls, st, ed, num_frame, indices, fps=3):
        """ Get temporal supervise signal
        Args:
            st (float):
            ed (float):
            num_frame (int):
            indices (np.ndarray): fps0.5 indices
            fps (int): frame rate used to extract the frames
        Returns:
            sup_ts_type==`st_ed`: [start_idx, end_idx]
        """
        max_num_frame = 300.
        if num_frame > max_num_frame:
            st, ed = [(max_num_frame / num_frame) * fps * ele for ele in [st, ed]]
        else:
            st, ed = [fps * ele for ele in [st, ed]]

        start_idx = np.searchsorted(indices, st, side="left")
        end_idx = np.searchsorted(indices, ed, side="right")
        max_len = len(indices)
        if not start_idx < max_len:
            start_idx -= 1
        if not end_idx < max_len:
            end_idx -= 1
        if start_idx == end_idx:
            st_ed = [start_idx, end_idx]
        else:
            st_ed = [start_idx, end_idx-1]  # this is the correct formula

        return st_ed  # (2, )

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        # !!!! remove comma here, since they are too many of them, !!! no removing  # TODO
        # words = [w for w in words if w != ","]
        words = [w for w in words]
        words = words + [eos_word] if eos else words
        return words

    @classmethod
    def find_match(cls, subtime, value, mode="larger", span=1.5):
        """closet value in an array to a given value"""
        if mode == "nearest":  # closet N samples
            return sorted((np.abs(subtime - value)).argsort()[:2].tolist())
        elif mode == "span":  # with a specified time span
            return_indices = np.nonzero(np.abs(subtime - value) < span)[0].tolist()
            if value <= 2:
                return_indices = np.nonzero(subtime - 2 <= 0)[0].tolist() + return_indices
            return return_indices
        elif mode == "larger":
            idx = max(0, np.searchsorted(subtime, value, side="left") - 1)
            return_indices = [idx - 1, idx, idx + 1]
            return_indices = [idx for idx in return_indices if 0 <= idx < len(subtime)]
            return return_indices

    @classmethod
    def get_aligned_sub_indices(cls, img_ids, subtext, subtime, fps=3, mode="larger"):
        """ Get aligned subtitle for each frame, for each frame, use the two subtitle
        sentences that are most close to it
        Args:
            img_ids (list of int): image file ids, note the image index starts from 1. Is one possible???
            subtext (str): tokenized subtitle sentences concatenated by "<eos>".
            subtime (list of float): a list of timestamps from the subtile file, each marks the start
                of each subtile sentence. It should have the same length as the "<eos>" splitted subtext.
            fps (int): frame per second when extracting the video
            mode (str): nearest or larger
        Returns:
            a list of str, each str should be aligned with an image indicated by img_ids.
        """
        subtext = subtext.split(" <eos> ")  # note the spaces
        raw_sub_n_tokens = [len(s.split()) for s in subtext]
        assert len(subtime) == len(subtext)
        img_timestamps = np.array(img_ids) / fps  # roughly get the timestamp for the
        img_aligned_sentence_indices = []  # list(list)
        for t in img_timestamps:
            img_aligned_sentence_indices.append(cls.find_match(subtime, t, mode=mode))
        return img_aligned_sentence_indices, raw_sub_n_tokens

    @classmethod
    def get_aligned_sub(cls, subtext, img_aligned_sentence_indices):
        subtext = subtext.split(" <eos> ")  # note the spaces
        return [" ".join([subtext[inner_idx] for inner_idx in e]) for e in img_aligned_sentence_indices]

    def mk_noun_mask(self, noun_indices_q, noun_indices_a, q_l, a_l, eos=True):
        """ mask is a ndarray (num_q_words + num_ca_words + 1, )
        removed nouns that are not in the vocabulary
        Args:
            noun_indices_q (list): each element is [index, word]
            noun_indices_a (list):
            q_l (int):
            a_l (int):
            eos

        Returns:

        """
        noun_indices_q = [e[0] for e in noun_indices_q if e[1].lower() in self.word2idx]
        noun_indices_a = [e[0] + q_l for e in noun_indices_a if e[1].lower() in self.word2idx]
        noun_indices = np.array(noun_indices_q + noun_indices_a) - 1
        mask = np.zeros(q_l + a_l + 1) if eos else np.zeros(q_l + a_l)
        if len(noun_indices) != 0:  # seems only 1 instance has no indices
            mask[noun_indices] = 1
        return mask

    @classmethod
    def get_labels_single_box(cls, single_box, detected_boxes):
        """return a list of IoUs"""
        gt_box = [single_box["left"], single_box["top"],
                  single_box["left"] + single_box["width"],
                  single_box["top"] + single_box["height"]]  # [left, top, right, bottom]
        IoUs = [float("{:.4f}".format(computeIoU(gt_box, d_box))) for d_box in detected_boxes]
        return IoUs

    def get_iou_data(self, gt_box_data_i, meta_data_i, frm_cnt_i):
        """
        meta_data (dict):  with vid_name as key,
        add iou_data entry, organized similar to bbox_data
        """
        frm_cnt_i = frm_cnt_i + 1  # add extra 1 since img_ids are 1-indexed
        iou_data_i = {}
        img_ids = sorted(gt_box_data_i.keys(), key=lambda x: int(x))
        img_ids = [e for e in img_ids if int(e) < frm_cnt_i]
        for img_id in img_ids:
            iou_data_i[img_id] = []
            cur_detected_boxes = meta_data_i["boxes"][int(img_id) - 1]
            for box in gt_box_data_i[img_id]:
                iou_list = self.get_labels_single_box(box, cur_detected_boxes)
                iou_data_i[img_id].append({
                    "iou": iou_list,
                    "label": box["label"],
                    "img_id": img_id
                })
        return iou_data_i

    @classmethod
    def mk_att_label(cls, iou_data, q_ca_sentence, region_cnts, ca_len, iou_thd=0.5, single_box=False):
        """return a list(dicts) of length num_imgs, each dict with word indices as keys,
        with corresponding region index as values.
        iou_data:
        q_ca_sentence: q(str) + " " + ca(str)
        region_cnts: list(int)
        ca_len: int, number of words for the concatenation of question the correct answer, +1 for eos
        single_box (bool): return a single object box for each gt box (the one with highest IoU)
        """
        img_ids = sorted(iou_data.keys(), key=lambda x: int(x))
        q_ca_words = q_ca_sentence.split()
        att_label = [np.zeros((ca_len, cnt)) for cnt in region_cnts]  # #imgs * (#words, #regions)
        for idx, img_id in enumerate(img_ids):  # within a single image
            cur_img_iou_info = iou_data[img_id]
            cur_labels = [e["label"] for e in cur_img_iou_info]  # might be upper case
            for noun_idx in range(ca_len-1):  # do not count <EOS> in
                # find the gt boxes (possibly > 1) under the same label
                cur_noun = q_ca_words[noun_idx]
                cur_box_indices = [box_idx for box_idx, label in enumerate(cur_labels)
                                   if label.lower() == cur_noun.lower()]

                # find object boxes that has high IoU with gt boxes, 1 or more for each gt box (single_box)
                cur_iou_mask = None
                for box_idx in cur_box_indices:
                    if cur_iou_mask is None:
                        # why is [:region_cnts[idx]] The cnt here is actually after min(cnt, max_num_regions)
                        if single_box:
                            cur_ios_mask_len = len(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]])
                            cur_iou_mask = np.zeros(cur_ios_mask_len)
                            if np.max(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd:
                                cur_iou_mask[np.argmax(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]])] = 1
                        else:
                            cur_iou_mask = np.array(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd
                    else:
                        if single_box:  # assume the high IoU boxes for the same label will not be the same
                            if np.max(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd:
                                cur_iou_mask[np.argmax(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]])] = 1
                        else:
                            # [True, False] + [True, True] = [True, True]
                            cur_iou_mask += np.array(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd
                if cur_iou_mask is not None:
                    # less than num_regions is possible,
                    # we assume the attention is evenly paid to overlapped boxes
                    if cur_iou_mask.sum() != 0:
                        cur_iou_mask = cur_iou_mask.astype(np.float32) / cur_iou_mask.sum()  # TODO
                    att_label[idx][noun_idx] = cur_iou_mask
        return [torch.from_numpy(e) for e in att_label]  # , att_label_mask

    def numericalize(self, sentence, eos=True, match=False):
        """convert words to indices, match stanford tokenizer"""
        if match:
            sentence = match_stanford_tokenizer(sentence)
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
        return sentence_indices

    def numericalize_hier_vcpt(self, vcpt_words_list):
        """vcpt_words_list is a list of sublist, each sublist contains words"""
        sentence_indices = []
        for i in range(len(vcpt_words_list)):
            # some labels are 'tennis court', keep the later word
            words = [e.split()[-1] for e in vcpt_words_list[i]]
            sentence_indices.append([self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                                     for w in words])
        return sentence_indices

    def numericalize_vcpt(self, vcpt_sentence):
        """convert words to indices, additionally removes duplicated attr-object pairs"""
        attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
        attr_obj_pairs = [e.strip() for e in attr_obj_pairs]
        unique_pairs = []
        for pair in attr_obj_pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        words.append("<eos>")
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in words]
        return sentence_indices


def pad_sequence_3d_label(sequences, sequences_masks):
    """
    Args:
        sequences: list(3d torch.Tensor)
        sequences_masks: list(torch.Tensor) of the same shape as sequences,
            individual mask is the result of masking the individual element

    Returns:

    """
    shapes = [seq.shape for seq in sequences]
    lengths_1 = [s[0] for s in shapes]
    lengths_2 = [s[1] for s in shapes]
    lengths_3 = [s[2] for s in shapes]
    padded_seqs = torch.zeros(len(sequences), max(lengths_1), max(lengths_2), max(lengths_3)).float()
    mask = copy.deepcopy(padded_seqs)
    for idx, seq in enumerate(sequences):
        padded_seqs[idx, :lengths_1[idx], :lengths_2[idx], :lengths_3[idx]] = seq
        mask[idx, :lengths_1[idx], :lengths_2[idx], :lengths_3[idx]] = sequences_masks[idx]
    return padded_seqs, mask


def pad_sequences_2d(sequences, dtype=torch.long):
    """ Pad a double-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first two dims has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases

    Returns:

    Examples:
        >>> test_data_list = [[[1, 3, 5], [3, 7, 4, 1]], [[98, 34, 11, 89, 90], [22], [34, 56]],]
        >>> pad_sequences_2d(test_data_list, dtype=torch.long)  # torch.Size([2, 3, 5])
        >>> test_data_3d = [torch.randn(2,2,4), torch.randn(4,3,4), torch.randn(1,5,4)]
        >>> pad_sequences_2d(test_data_3d, dtype=torch.float)  # torch.Size([2, 3, 5])
        >>> test_data_3d2 = [[torch.randn(2,4), ], [torch.randn(3,4), torch.randn(5,4)]]
        >>> pad_sequences_2d(test_data_3d2, dtype=torch.float)  # torch.Size([2, 3, 5])
    """
    bsz = len(sequences)
    para_lengths = [len(seq) for seq in sequences]
    max_para_len = max(para_lengths)
    sen_lengths = [[len(word_seq) for word_seq in seq] for seq in sequences]
    max_sen_len = max(flat_list_of_lists(sen_lengths))

    if isinstance(sequences[0], torch.Tensor):
        extra_dims = sequences[0].shape[2:]
    elif isinstance(sequences[0][0], torch.Tensor):
        extra_dims = sequences[0][0].shape[1:]
    else:
        sequences = [[torch.LongTensor(word_seq) for word_seq in seq] for seq in sequences]
        extra_dims = ()

    padded_seqs = torch.zeros((bsz, max_para_len, max_sen_len) + extra_dims, dtype=dtype)
    mask = torch.zeros(bsz, max_para_len, max_sen_len).float()

    for b_i in range(bsz):
        for sen_i, sen_l in enumerate(sen_lengths[b_i]):
            padded_seqs[b_i, sen_i, :sen_l] = sequences[b_i][sen_i]
            mask[b_i, sen_i, :sen_l] = 1
    return padded_seqs, mask  # , sen_lengths


def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def make_mask_from_length(lengths):
    mask = torch.zeros(len(lengths), max(lengths)).float()
    for idx, l in enumerate(lengths):
        mask[idx, :l] = 1
    return mask


class PadCollate:
    def __init__(self, hparams):
        self.max_len_dict = dict(
            max_sub_l=hparams.max_sub_l,
            max_vid_l=hparams.max_vid_l,
            max_qa_l=hparams.max_qa_l,
            max_num_regions=hparams.num_region
        )

    def limit_len(self, batch):
        # qas (B, 5, #words, D)
        max_qa_l = min(batch["qas"].shape[2], self.max_len_dict["max_qa_l"])
        batch["qas"] = batch["qas"][:, :, :max_qa_l]
        batch["qas_bert"] = batch["qas_bert"][:, :, :max_qa_l]
        batch["qas_mask"] = batch["qas_mask"][:, :, :max_qa_l]

        # (B, #imgs, #words, D)
        batch["sub"] = batch["sub"][:, :self.max_len_dict["max_vid_l"], :self.max_len_dict["max_sub_l"]]
        batch["sub_bert"] = batch["sub_bert"][:, :self.max_len_dict["max_vid_l"], :self.max_len_dict["max_sub_l"]]
        batch["sub_mask"] = batch["sub_mask"][:, :self.max_len_dict["max_vid_l"], :self.max_len_dict["max_sub_l"]]

        # context, vid (B, #imgs, #regions, D), vcpt (B, #imgs, #regions)
        ctx_keys = ["vid"]
        for k in ctx_keys:
            max_l = min(batch[k].shape[1], self.max_len_dict["max_{}_l".format(k)])
            batch[k] = batch[k][:, :max_l]
            mask_key = "{}_mask".format(k)
            batch[mask_key] = batch[mask_key][:, :max_l]

        # att_label (B, #imgs, #qa_words, #regions)
        if batch["att_labels"] is not None:
            batch["att_labels"] = batch["att_labels"][:, :self.max_len_dict["max_vid_l"],
                                  :self.max_len_dict["max_qa_l"], :self.max_len_dict["max_num_regions"]]
            batch["att_labels_mask"] = batch["att_labels_mask"][:, :self.max_len_dict["max_vid_l"],
                                       :self.max_len_dict["max_qa_l"], :self.max_len_dict["max_num_regions"]]
        # batch["anno_st_idx"] = batch["anno_st_idx"]

        if batch["ts_label"] is None:
            batch["ts_label"] = None
            batch["ts_label_mask"] = None
        elif isinstance(batch["ts_label"], dict):  # (st_ed, ce)
            batch["ts_label"] = dict(
                st=batch["ts_label"]["st"],
                ed=batch["ts_label"]["ed"]
            )
            batch["ts_label_mask"] = batch["ts_label_mask"][:, :self.max_len_dict["max_vid_l"]]
        else:  # frm-wise or (st_ed, bce)
            batch["ts_label"] = batch["ts_label"][:, :self.max_len_dict["max_vid_l"]]
            batch["ts_label_mask"] = batch["ts_label_mask"][:, :self.max_len_dict["max_vid_l"]]

        # target
        # batch["target"] = batch["target"]

        # others
        # batch["qid"] = batch["qid"]
        # model_in_dict["vid_name"] = batch["vid_name"]
        batch["vid_name"] = None

        # model_in_dict["ts"] = batch["ts"]  # $%#$%@#$^@#$^@$^?W?S?DFS?DF
        # batch["q_l"] = batch["q_l"]
        # batch["image_indices"] = batch["image_indices"]
        # batch["image_indices_mask"] = batch["image_indices_mask"]

        # if batch["boxes"].dim() == 1:
        #     batch["boxes"] = batch["boxes"]
        #     batch["boxes_mask"] = batch["boxes"]
        # else:
        #     batch["boxes"] = batch["boxes"][:, :self.max_len_dict["max_vid_l"], :self.max_len_dict["max_num_regions"], :]
        #     batch["boxes_mask"] = batch["boxes_mask"][:, :self.max_len_dict["max_vid_l"], :self.max_len_dict["max_num_regions"]]

        return batch

    def pad_collate(self, data):
        # separate source and target sequences
        batch = dict()

        batch["qas"], batch["qas_mask"] = pad_sequences_2d([d["qas"] for d in data], dtype=torch.long)
        batch["qas_bert"], _ = pad_sequences_2d([d["qas_bert"] for d in data], dtype=torch.float)
        batch["sub"], batch["sub_mask"] = pad_sequences_2d([d["sub"] for d in data], dtype=torch.long)
        batch["sub_bert"], _ = pad_sequences_2d([d["sub_bert"] for d in data], dtype=torch.float)
        batch["vid_name"] = [d["vid_name"] for d in data]  # inference
        batch["qid"] = torch.LongTensor([d["qid"] for d in data])  # inference
        batch["target"] = torch.tensor([d["target"] for d in data], dtype=torch.long)
        batch["vid"], batch["vid_mask"] = pad_sequences_2d([d["vfeat"] for d in data], dtype=torch.float)

        if data[0]["att_labels"] is None:
            batch["att_labels"] = None
        else:
            batch["att_labels"], batch["att_labels_mask"] = pad_seq_of_seq_of_tensors([d["att_labels"] for d in data])
        batch["anno_st_idx"] = torch.LongTensor([d["anno_st_idx"] for d in data])  # list(int) # $$$$
        if data[0]["ts_label"] is None:
            batch["ts_label"] = None
        elif isinstance(data[0]["ts_label"], list):  # (st_ed, ce)
            batch["ts_label"] = dict(
                st=torch.LongTensor([d["ts_label"][0] for d in data]),
                ed=torch.LongTensor([d["ts_label"][1] for d in data]),
            )
            batch["ts_label_mask"] = make_mask_from_length([len(d["image_indices"]) for d in data])
        elif isinstance(data[0]["ts_label"], torch.Tensor):  # (st_ed, bce) or frm
            batch["ts_label"], batch["ts_label_mask"] = pad_sequences_1d([d["ts_label"] for d in data],
                                                                         dtype=torch.float)
        else:
            raise NotImplementedError

        batch["image_indices"], batch["image_indices_mask"] = pad_sequences_1d([d["image_indices"] for d in data],
                                                                               dtype=torch.long)
        batch["q_l"] = torch.LongTensor([d["q_l"] for d in data])

        # if data[0]["boxes"] is not None:
        #     batch["boxes"], batch["boxes_mask"] = pad_boxes_nested_sequences([d["boxes"] for d in data])
        # else:
        #     batch["boxes"] = np.array([d["boxes"] for d in data], dtype=float)
        #     batch["boxes"] = torch.tensor(batch["boxes"], dtype=torch.float)
        #     batch["boxes_mask"] = batch["boxes"]
        batch["boxes_mask"] = batch["boxes"] = None

        batch = self.limit_len(batch)

        return batch

    def __call__(self, data):
        return self.pad_collate(data)


def pad_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    """
    # separate source and target sequences
    batch = dict()
    batch["qas"], batch["qas_mask"] = pad_sequences_2d([d["qas"] for d in data], dtype=torch.long)
    batch["qas_bert"], _ = pad_sequences_2d([d["qas_bert"] for d in data], dtype=torch.float)
    batch["sub"], batch["sub_mask"] = pad_sequences_2d([d["sub"] for d in data], dtype=torch.long)
    batch["sub_bert"], _ = pad_sequences_2d([d["sub_bert"] for d in data], dtype=torch.float)
    batch["vid_name"] = [d["vid_name"] for d in data]  # inference
    batch["qid"] = torch.LongTensor([d["qid"] for d in data])  # inference
    batch["target"] = torch.tensor([d["target"] for d in data], dtype=torch.long)
    batch["vid"], batch["vid_mask"] = pad_sequences_2d([d["vfeat"] for d in data], dtype=torch.float)

    if data[0]["att_labels"] is None:
        batch["att_labels"] = None
    else:
        batch["att_labels"], batch["att_labels_mask"] = pad_seq_of_seq_of_tensors([d["att_labels"] for d in data])
    batch["anno_st_idx"] = torch.LongTensor([d["anno_st_idx"] for d in data])  # list(int) # $$$$
    if data[0]["ts_label"] is None:
        batch["ts_label"] = None
    elif isinstance(data[0]["ts_label"], list):  # (st_ed, ce)
        batch["ts_label"] = dict(
            st=torch.LongTensor([d["ts_label"][0] for d in data]),
            ed=torch.LongTensor([d["ts_label"][1] for d in data]),
        )
        batch["ts_label_mask"] = make_mask_from_length([len(d["image_indices"]) for d in data])
    elif isinstance(data[0]["ts_label"], torch.Tensor):  # (st_ed, bce) or frm
        batch["ts_label"], batch["ts_label_mask"] = pad_sequences_1d([d["ts_label"] for d in data], dtype=torch.float)
    else:
        raise NotImplementedError

    batch["image_indices"], batch["image_indices_mask"] = pad_sequences_1d([d["image_indices"] for d in data], dtype=torch.long)
    batch["q_l"] = torch.LongTensor([d["q_l"] for d in data])

    # if data[0]["boxes"] is not None:
    #     batch["boxes"], batch["boxes_mask"] = pad_boxes_nested_sequences([d["boxes"] for d in data])
    # else:
    #     batch["boxes"] = np.array([d["boxes"] for d in data], dtype=float)
    #     batch["boxes"] = torch.tensor(batch["boxes"], dtype=torch.float)
    #     batch["boxes_mask"] = batch["boxes"]
    batch["boxes_mask"] = batch["boxes"] = None

    return batch


def prepare_inputs(batch, max_len_dict=None, device="cuda", non_blocking=True):
    """clip and move input data to gpu"""
    model_in_dict = dict()

    # qas (B, 5, #words, D)
    max_qa_l = min(batch["qas"].shape[2], max_len_dict["max_qa_l"])
    model_in_dict["qas"] = batch["qas"][:, :, :max_qa_l].to(device=device, non_blocking=non_blocking)
    model_in_dict["qas_bert"] = batch["qas_bert"][:, :, :max_qa_l].to(device=device, non_blocking=non_blocking)
    model_in_dict["qas_mask"] = batch["qas_mask"][:, :, :max_qa_l].to(device=device, non_blocking=non_blocking)

    # (B, #imgs, #words, D)
    model_in_dict["sub"] = batch["sub"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device=device, non_blocking=non_blocking)
    model_in_dict["sub_bert"] = batch["sub_bert"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device=device, non_blocking=non_blocking)
    model_in_dict["sub_mask"] = batch["sub_mask"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device=device, non_blocking=non_blocking)

    # context, vid (B, #imgs, #regions, D), vcpt (B, #imgs, #regions)
    ctx_keys = ["vid", "vcpt"]
    for k in ctx_keys:
        max_l = min(batch[k].shape[1], max_len_dict["max_{}_l".format(k)])
        model_in_dict[k] = batch[k][:, :max_l].to(device=device, non_blocking=non_blocking)
        mask_key = "{}_mask".format(k)
        model_in_dict[mask_key] = batch[mask_key][:, :max_l].to(device=device, non_blocking=non_blocking)

    # att_label (B, #imgs, #qa_words, #regions)
    if batch["att_labels"] is not None:
        model_in_dict["att_labels"] = batch["att_labels"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_qa_l"],
                                                          :max_len_dict["max_num_regions"]].to(device=device,
                                                                                               non_blocking=non_blocking)
        model_in_dict["att_labels_mask"] = batch["att_labels_mask"][:, :max_len_dict["max_vid_l"],
                                                                    :max_len_dict["max_qa_l"],
                                                                    :max_len_dict["max_num_regions"]].to(device=device,
                                                                                                         non_blocking=non_blocking)
    model_in_dict["anno_st_idx"] = batch["anno_st_idx"].to(device=device, non_blocking=non_blocking)

    if batch["ts_label"] is None:
        model_in_dict["ts_label"] = None
        model_in_dict["ts_label_mask"] = None
    elif isinstance(batch["ts_label"], dict):  # (st_ed, ce)
        model_in_dict["ts_label"] = dict(
            st=batch["ts_label"]["st"].to(device=device, non_blocking=non_blocking),
            ed=batch["ts_label"]["ed"].to(device=device, non_blocking=non_blocking),
        )
        model_in_dict["ts_label_mask"] = batch["ts_label_mask"][:, :max_len_dict["max_vid_l"]].to(device=device, non_blocking=non_blocking)
    else:  # frm-wise or (st_ed, bce)
        model_in_dict["ts_label"] = batch["ts_label"][:, :max_len_dict["max_vid_l"]].to(device=device, non_blocking=True)
        model_in_dict["ts_label_mask"] = batch["ts_label_mask"][:, :max_len_dict["max_vid_l"]].to(device=device, non_blocking=non_blocking)

    # target
    model_in_dict["target"] = batch["target"].to(device=device, non_blocking=non_blocking)

    # others
    model_in_dict["qid"] = batch["qid"].to(device=device, non_blocking=non_blocking)
    # model_in_dict["vid_name"] = batch["vid_name"]
    model_in_dict["vid_name"] = None

    targets = model_in_dict["target"]
    qids = model_in_dict["qid"]
    # model_in_dict["ts"] = batch["ts"]  # $%#$%@#$^@#$^@$^?W?S?DFS?DF
    model_in_dict["q_l"] = batch["q_l"].to(device=device, non_blocking=non_blocking)  # $%#$%@#$^@#$^@$^?W?S?DFS?DF
    model_in_dict["image_indices"] = batch["image_indices"].to(device=device, non_blocking=non_blocking)
    model_in_dict["image_indices_mask"] = batch["image_indices_mask"].to(device=device, non_blocking=non_blocking)
    if batch["boxes"].dim() == 1:
        model_in_dict["boxes"] = batch["boxes"]
        model_in_dict["boxes_mask"] = batch["boxes"]
    else:
        model_in_dict["boxes"] = batch["boxes"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_num_regions"], :].to(device=device, non_blocking=non_blocking)
        model_in_dict["boxes_mask"] = batch["boxes_mask"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_num_regions"]].to(device=device, non_blocking=non_blocking)

    # model_in_dict["object_labels"] = batch["object_labels"]  # $%#$%@#$^@#$^@$^?W?S?DFS?DF
    return model_in_dict, targets, qids


def find_match(subtime, time_array, mode="larger"):
    """find closet value in an array to a given value
    subtime (float):
    time_array (np.ndarray): (N, )
    """
    if mode == "nearest":
        return (np.abs(subtime - time_array)).argsort()[:2].tolist()
    elif mode == "larger":
        idx = max(0, np.searchsorted(subtime, time_array, side="left") - 1)
        return_indices = [idx-1, idx, idx+1]
        # return_indices = [idx, idx+1]
        return_indices = [idx for idx in return_indices if 0 <= idx < len(subtime)]
        return return_indices
    else:
        raise NotImplementedError


def pad_seq_of_seq_of_tensors(sequences, dtype=torch.long):
    # att_label (B, #imgs, #qa_words, #regions)

    bsz = len(sequences)
    num_imgs = [len(seq) for seq in sequences]
    max_imgs = max(num_imgs)
    img_shapes = [[tuple(img.shape) for img in seq] for seq in sequences]
    num_words, num_regions = list(zip(*flat_list_of_lists(img_shapes)))
    max_words = max(num_words)
    max_regions = max(num_regions)

    padded_seqs = torch.zeros((bsz, max_imgs, max_words, max_regions), dtype=dtype)
    mask = torch.zeros(bsz, max_imgs, max_words, max_regions).float()

    for b_i in range(bsz):
        for img in range(num_imgs[b_i]):
            n_words, n_regions = img_shapes[b_i][img]
            padded_seqs[b_i, img, :n_words, :n_regions] = sequences[b_i][img]
            mask[b_i, img, :n_words, :n_regions] = 1
    return padded_seqs, mask  # , sen_lengths


def pad_boxes_nested_sequences(sequences, dtype=torch.float):
    bsz = len(sequences)
    num_imgs = [len(sample) for sample in sequences]
    num_boxes = [[len(img) for img in sample] for sample in sequences]
    num_coordinates = 4

    max_imgs = max(num_imgs)
    max_boxes = max(flat_list_of_lists(num_boxes))

    padded_seqs = torch.zeros((bsz, max_imgs, max_boxes, num_coordinates), dtype=dtype)
    mask = torch.zeros(bsz, max_imgs, max_boxes).float()

    for b_i in range(bsz):
        for img in range(num_imgs[b_i]):
            for box in range(num_boxes[b_i][img]):
                padded_seqs[b_i, img, box] = torch.Tensor(sequences[b_i][img][box])
                mask[b_i, img, box] = 1

    return padded_seqs, mask
