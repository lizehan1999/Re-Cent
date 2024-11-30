from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.utils import ModelOutput

from .encoder import Encoder
from .layers import create_projection_layer, SoRepLayer
from .loss_functions import focal_loss_with_logits
from .span_rep import SpanRepLayer

torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=1000)


@dataclass
class ReCentModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[List[torch.FloatTensor]] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    span_rep_so: Optional[List[torch.FloatTensor]] = None


def extract_prompt_features_and_word_embeddings(config, token_embeds, input_ids, attention_mask,
                                                text_lengths, words_mask, **kwargs):
    # getting prompt embeddings
    batch_size, sequence_length, embed_dim = token_embeds.shape

    class_token_mask = input_ids == config.class_token_index
    num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)
    max_embed_dim = num_class_tokens.max()
    max_text_length = text_lengths.max()
    aranged_class_idx = torch.arange(max_embed_dim,
                                     dtype=attention_mask.dtype,
                                     device=token_embeds.device).expand(batch_size, -1)

    batch_indices, target_class_idx = torch.where(aranged_class_idx < num_class_tokens)
    _, class_indices = torch.where(class_token_mask)

    prompts_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    prompts_embedding_mask = (aranged_class_idx < num_class_tokens).to(attention_mask.dtype)

    prompts_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]

    # getting words embedding
    words_embedding = torch.zeros(
        batch_size, max_text_length, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    batch_indices, word_idx = torch.where(words_mask > 0)

    target_word_idx = words_mask[batch_indices, word_idx] - 1

    words_embedding[batch_indices, target_word_idx] = token_embeds[batch_indices, word_idx]

    aranged_word_idx = torch.arange(max_text_length,
                                    dtype=attention_mask.dtype,
                                    device=token_embeds.device).expand(batch_size, -1)

    mask = aranged_word_idx < text_lengths

    return prompts_embedding, prompts_embedding_mask, words_embedding, mask


class BaseModel(ABC, nn.Module):
    def __init__(self, config, from_pretrained=False):
        super(BaseModel, self).__init__()
        self.config = config

        self.token_rep_layer = Encoder(config, from_pretrained)

    def _extract_prompt_features_and_word_embeddings(self, token_embeds, input_ids, attention_mask,
                                                     text_lengths, words_mask):
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = extract_prompt_features_and_word_embeddings(
            self.config,
            token_embeds,
            input_ids,
            attention_mask,
            text_lengths,
            words_mask)
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def get_representations(self,
                            input_ids: Optional[torch.FloatTensor] = None,
                            attention_mask: Optional[torch.LongTensor] = None,
                            text_lengths: Optional[torch.Tensor] = None,
                            words_mask: Optional[torch.LongTensor] = None,
                            **kwargs):
        token_embeds = self.token_rep_layer(input_ids, attention_mask, **kwargs)

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self._extract_prompt_features_and_word_embeddings(
            token_embeds, input_ids, attention_mask,
            text_lengths, words_mask)
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    @abstractmethod
    def forward(self, x):
        pass

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor,
              alpha: float = -1., gamma: float = 0.0):
        all_losses = focal_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma)
        return all_losses

    @abstractmethod
    def loss(self, x):
        pass


class SpanModel(BaseModel):
    def __init__(self, config, encoder_from_pretrained):
        super(SpanModel, self).__init__(config, encoder_from_pretrained)
        self.span_rep_layer_sub = SpanRepLayer(
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout)
        self.span_rep_layer_ob = SpanRepLayer(
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout)

        self.so_rep_layer = SoRepLayer(hidden_size=config.hidden_size,
                                       dropout=config.dropout)

        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)

        self.beta = config.loss_beta  # loss control
        self.so_neg_num = config.so_neg_num
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self,
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                span_idx: Optional[torch.LongTensor] = None,
                span_mask_sub: Optional[torch.LongTensor] = None,
                span_mask_ob: Optional[torch.LongTensor] = None,
                labels_sub: Optional[torch.FloatTensor] = None,
                labels_ob: Optional[torch.FloatTensor] = None,
                sub_ob_pair: Optional[Dict] = None,
                if_reverse=False,
                **kwargs
                ):

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(input_ids,
                                                                                                    attention_mask,
                                                                                                    text_lengths,
                                                                                                    words_mask)

        span_idx_sub = span_idx * span_mask_sub.unsqueeze(-1)
        span_idx_ob = span_idx * span_mask_ob.unsqueeze(-1)

        span_rep_sub = self.span_rep_layer_sub(words_embedding, span_idx_sub)
        span_rep_ob = self.span_rep_layer_ob(words_embedding, span_idx_ob)

        prompts_embedding = self.prompt_rep_layer(prompts_embedding)

        scores_sub = torch.einsum("BLKD,BCD->BLKC", span_rep_sub, prompts_embedding)
        scores_ob = torch.einsum("BLKD,BCD->BLKC", span_rep_ob, prompts_embedding)

        bs, _, span_len, num_classes = scores_sub.shape

        scores_sub_flat = scores_sub.reshape(bs, -1, scores_sub.shape[-1])
        scores_ob_flat = scores_ob.reshape(bs, -1, scores_ob.shape[-1])

        topk_indices_sub, topk_indices_ob, gold_so_labels_all, topk_mask_so = self.find_topk_so(scores_sub_flat,
                                                                                                scores_ob_flat,
                                                                                                sub_ob_pair, bs,
                                                                                                span_len)

        so_reps, so_labels = self.get_so_matrix(topk_indices_sub, topk_indices_ob, span_rep_sub, span_rep_ob,
                                                gold_so_labels_all, num_classes)
        scores_so = torch.einsum("BSOD,BCD->BSOC", so_reps, prompts_embedding)

        scores = [scores_sub, scores_ob]

        loss = None
        if labels_sub is not None and labels_ob is not None:
            loss_sub = self.loss(scores_sub, labels_sub, prompts_embedding_mask, span_mask_sub, **kwargs)
            loss_ob = self.loss(scores_ob, labels_ob, prompts_embedding_mask, span_mask_ob, **kwargs)
            loss_so = self.loss(scores_so, so_labels, prompts_embedding_mask, topk_mask_so, **kwargs)
            loss = self.beta * (loss_sub + loss_ob) + (1 - self.beta) * loss_so

        output = ReCentModelOutput(
            logits=scores,  # BLKC
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
            span_rep_so=[span_rep_sub, span_rep_ob]  # BLKD
        )
        return output

    def loss(self, scores, labels, prompts_embedding_mask, mask_label,
             alpha: float = -1., gamma: float = 0.0, **kwargs):

        batch_size = scores.shape[0]
        num_classes = scores.shape[-1]

        scores = scores.view(-1, num_classes)
        labels = labels.view(-1, num_classes)

        all_losses = self._loss(scores, labels, alpha, gamma)

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.view(-1, 1)

        all_losses = all_losses * mask_label.float()

        loss = all_losses.sum()
        return loss

    def subject_object_correlation(self, rep_sub_selected, rep_ob_selected, prompts_embedding):
        so_reps = self.so_rep_layer(rep_sub_selected, rep_ob_selected)
        scores_so = torch.einsum("BSOD,BCD->BSOC", so_reps, prompts_embedding)
        scores_so = torch.sigmoid(scores_so.squeeze())
        return scores_so

    def find_topk_so(self, scores_sub_flat, scores_ob_flat, sub_ob_pair, bs, span_len):
        topk_values_sub, topk_indices_sub = torch.topk(scores_sub_flat, k=self.so_neg_num, dim=1, largest=True,
                                                       sorted=False)  # [bs,topk,class]
        topk_values_ob, topk_indices_ob = torch.topk(scores_ob_flat, k=self.so_neg_num, dim=1, largest=True,
                                                     sorted=False)  # [bs,topk,class]
        topk_indices_sub = topk_indices_sub.reshape(bs, -1)
        topk_indices_ob = topk_indices_ob.reshape(bs, -1)

        gold_sub_all, gold_ob_all, gold_so_labels_all = [], [], []
        for sub_ob in sub_ob_pair:
            gold_sub, gold_ob, gold_so_labels = [], [], {}
            for so in sub_ob.keys():
                sub_id = so[0][0] * (span_len - 1) + so[0][1]
                ob_id = so[1][0] * (span_len - 1) + so[1][1]
                gold_sub.append(sub_id)
                gold_ob.append(ob_id)
                gold_so_labels[(sub_id, ob_id)] = sub_ob[so]
            gold_sub_all.append(gold_sub)
            gold_ob_all.append(gold_ob)
            gold_so_labels_all.append(gold_so_labels)

        gold_sub_all = self.padding_ids(gold_sub_all)
        gold_ob_all = self.padding_ids(gold_ob_all)
        topk_indices_sub = torch.cat((topk_indices_sub, gold_sub_all), dim=-1)
        topk_indices_ob = torch.cat((topk_indices_ob, gold_ob_all), dim=-1)
        topk_indices_sub, topk_mask_sub = self.dedup_and_pad(topk_indices_sub)
        topk_indices_ob, topk_mask_ob = self.dedup_and_pad(topk_indices_ob)

        topk_mask_so = torch.einsum('ij,ik->ijk', topk_mask_sub, topk_mask_ob)
        topk_mask_so = topk_mask_so.reshape(bs, -1)

        return topk_indices_sub, topk_indices_ob, gold_so_labels_all, topk_mask_so

    def padding_ids(self, gold_all):
        max_length = max(len(lst) for lst in gold_all)
        padded_list = [lst + [0] * (max_length - len(lst)) for lst in gold_all]
        tensor = torch.tensor(padded_list, device=self.device)
        return tensor

    def dedup_and_pad(self, tensor):
        batch_size, seq_len = tensor.size()
        result_list = []
        for i in range(batch_size):
            batch = tensor[i]
            unique_batch = torch.unique(batch, sorted=False)
            result_list.append(unique_batch)
        max_len = max(len(x) for x in result_list)
        padded_tensor = torch.zeros((batch_size, max_len), dtype=tensor.dtype, device=self.device)
        mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            padded_tensor[i, :len(result_list[i])] = result_list[i]
            mask_tensor[i, :len(result_list[i])] = True
        return padded_tensor, mask_tensor

    def get_so_matrix(self, indices_sub, indices_ob, rep_sub, rep_ob, gold_so_labels, num_classes):

        '''

        Args:
            indices_sub: [bs,indices_sub_len]
            indices_ob: [bs,indices_ob_len]
            rep_sub: [bs,len,12,D]
            rep_ob: [bs,len,12,D]
            gold_so_labels:[{(157, 84): 2, (157, 108): 3},{(205, 60): 2},...]
            num_classes:int
        Returns:
            so_reps: [bs,indices_sub_len,indices_ob_len,D]
            so_labels:[bs,indices_sub_len,indices_ob,num_classes]
        '''
        B, _, _, D = rep_sub.size()

        rep_sub = rep_sub.reshape(B, -1, D)  # [bs,total_len,D]
        rep_ob = rep_ob.reshape(B, -1, D)  # [bs,total_len,D]

        indices_sub_expanded = indices_sub.unsqueeze(-1).expand(-1, -1, D).long()  # [bs, indices_sub_len, D]
        rep_sub_selected = torch.gather(rep_sub, 1, indices_sub_expanded)  # [bs, indices_sub_len, D]

        indices_ob_expanded = indices_ob.unsqueeze(-1).expand(-1, -1, D).long()  # [bs, indices_ob_len, D]
        rep_ob_selected = torch.gather(rep_ob, 1, indices_ob_expanded)  # [bs, indices_ob_len, D]

        so_reps = self.so_rep_layer(rep_sub_selected, rep_ob_selected)

        so_labels = torch.zeros((B, indices_sub.shape[-1], indices_ob.shape[-1], num_classes), device=self.device)

        for i in range(B):
            for (sub_idx, ob_idx), label in gold_so_labels[i].items():
                sub_idx = (indices_sub[i] == sub_idx).nonzero(as_tuple=True)[0]
                ob_idx = (indices_ob[i] == ob_idx).nonzero(as_tuple=True)[0]
                so_labels[i, sub_idx[:, None], ob_idx[:, None], label - 1] = 1

        return so_reps, so_labels
