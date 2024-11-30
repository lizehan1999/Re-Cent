from abc import ABC, abstractmethod
from functools import partial
import torch

from .utils import has_overlapping, has_overlapping_nested


class BaseDecoder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def decode(self):
        pass

    def greedy_search(self, spans, flat_ner=True, multi_label=False):
        if flat_ner:
            has_ov = partial(has_overlapping, multi_label=multi_label)
        else:
            has_ov = partial(has_overlapping_nested, multi_label=multi_label)

        new_list = []
        span_prob = sorted(spans, key=lambda x: -x[-1])

        for i in range(len(spans)):
            b = span_prob[i]
            flag = False
            for new in new_list:
                if has_ov(b[:-1], new):
                    flag = True
                    break
            if not flag:
                new_list.append(b)

        new_list = sorted(new_list, key=lambda x: x[0])
        return new_list


class SpanDecoder(BaseDecoder):
    def decode(self, tokens, id_to_classes, model_output, model_output_rep, flat_ner=False, threshold=0.5,
               multi_label=False):
        probs = torch.sigmoid(model_output)
        spans = []
        rep_selected = []
        for i, _ in enumerate(tokens):
            probs_i = probs[i]
            rep_i = model_output_rep[i]
            wh_i = [i.tolist() for i in torch.where(probs_i > threshold)]
            span_i = []
            for s, k, c in zip(*wh_i):
                if s + k < len(tokens[i]):
                    span_i.append((s, s + k, id_to_classes[c + 1], probs_i[s, k, c].item()))

            # [(0, 1, 'military branch', 0.9943768382072449), (17, 18, 'military branch', 0.49076592922210693)]
            span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)

            rep = [rep_i[span[0], span[1] - span[0]] for span in span_i]

            spans.append(span_i)
            rep_selected.append(rep)
        return spans, rep_selected
