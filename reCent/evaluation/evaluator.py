from collections import defaultdict

import numpy as np
import torch
from seqeval.metrics.v1 import _prf_divide

def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b

# For RC
def compute_macro_PRF(
        predicted_idx: np.ndarray, gold_idx: np.ndarray, i=-1, empty_label=None
):
    # https://github.com/dinobby/ZS-BERT/blob/master/model/evaluation.py
    """
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    """
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = predicted_idx[:i] == r
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0.0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def score(pred_sents: list, gold_sents: list) -> dict:
    '''

    Args:
        pred_sents: [[{"head":"A","tail":"B","label":"C"},{"head":"D","tail":"E","label":"F"}],[{"head":"A","tail":"B","label":"C"}]]
        gold_sents: [[{"head":"A","tail":"B","label":"C"}],[{"head":"A","tail":"B","label":"C"}],[{"head":"A","tail":"B","label":"C"}]]
    Returns:

    '''
    assert len(pred_sents) == len(gold_sents)
    num_pred = 0
    num_gold = 0
    num_correct = 0

    for i in range(len(gold_sents)):
        num_pred += len(pred_sents[i])
        num_gold += len(gold_sents[i])
        for p in pred_sents[i]:
            for g in gold_sents[i]:
                if (p["head"], p["tail"], p["label"]) == (g["head"], g["tail"], g["label"]):
                    num_correct += 1

    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)

    info = dict(
        pred=pred_sents,
        gold=gold_sents,
        precision=str(precision),
        recall=str(recall),
        f1=str(safe_divide(2 * precision * recall, precision + recall)),
    )
    return info

def extract_tp_actual_correct(y_true, y_pred):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for type_name, (start, end), idx in y_true:
        entities_true[type_name].add((start, end, idx))
    for type_name, (start, end), idx in y_pred:
        entities_pred[type_name].add((start, end, idx))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum, target_names


def flatten_for_eval(y_true, y_pred):
    all_true = []
    all_pred = []

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend([t + [i] for t in true])
        all_pred.extend([p + [i] for p in pred])

    return all_true, all_pred


def compute_prf(y_true, y_pred, average='micro'):
    y_true, y_pred = flatten_for_eval(y_true, y_pred)

    pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(y_true, y_pred)

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=('precision', 'recall', 'f-score'),
        zero_division='warn'
    )

    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=('precision', 'recall', 'f-score'),
        zero_division='warn'
    )

    denominator = precision + recall
    denominator[denominator == 0.] = 1
    f_score = 2 * (precision * recall) / denominator

    return {'precision': precision[0], 'recall': recall[0], 'f_score': f_score[0]}


class Evaluator:
    def __init__(self, all_true, all_outs):
        self.all_true = all_true
        self.all_outs = all_outs

    def get_entities_fr(self, ents):
        all_ents = []
        for s, e, lab in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def get_entities_pr(self, ents):
        all_ents = []
        for s, e, lab, _ in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def transform_data(self):
        all_true_ent = []
        all_outs_ent = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_entities_fr(i)
            all_true_ent.append(e)
            e = self.get_entities_pr(j)
            all_outs_ent.append(e)
        return all_true_ent, all_outs_ent

    @torch.no_grad()
    def evaluate(self):
        all_true_typed, all_outs_typed = self.transform_data()
        precision, recall, f1 = compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1


def ent2triplet(all_sub, all_ob, all_reps_sub=None, all_reps_ob=None, model=None, prompts_embedding=None,
                classes_to_id=None, gamma=None, is_pred=True):
    # {"head": "A", "tail": "B", "label": "C"}'
    all_triplets = []
    top_triplets = []
    if is_pred:
        assert len(all_sub) == len(prompts_embedding)
        for s_list, o_list, s_rep_list, o_rep_list, p_rep_list in zip(all_sub, all_ob, all_reps_sub, all_reps_ob,
                                                                      prompts_embedding):
            triplets = []
            assert len(s_list) == len(s_rep_list)
            assert len(o_list) == len(o_rep_list)
            for s, s_r in zip(s_list, s_rep_list):
                for o, o_r in zip(o_list, o_rep_list):
                    if s[2] == o[2]:
                        s_r = s_r.squeeze().unsqueeze(0).unsqueeze(0)
                        o_r = o_r.squeeze().unsqueeze(0).unsqueeze(0)
                        p_rep_list = p_rep_list.squeeze().unsqueeze(0)
                        score_so = model.subject_object_correlation(s_r, o_r, p_rep_list)
                        correlation_score = score_so[int(classes_to_id[s[2]]) - 1].item()
                        triplet = {"head": (s[0], s[1]), "tail": (o[0], o[1]), "label": s[2],
                                   "logit": s[3] + o[3] + correlation_score}
                        triplets.append(triplet)

            multi_triplets = [t for t in triplets if t["logit"] > gamma]
            top_triplet = [max(triplets, key=lambda t: t["logit"])] if triplets else []
            all_triplets.append(multi_triplets)
            top_triplets.append(top_triplet)
    else:
        for s_list, o_list in zip(all_sub, all_ob):
            triplets = []
            for s, o in zip(s_list, o_list):
                assert s[2] == o[2]
                triplet = {"head": (s[0], s[1]), "tail": (o[0], o[1]), "label": s[2]}
                triplets.append(triplet)
            all_triplets.append(triplets)

    return all_triplets, top_triplets


def evaluate_single_multi(all_true_sub, all_outs_sub, all_true_ob, all_outs_ob, all_reps_sub, all_reps_ob, model,
                          prompts_embedding, classes_to_id, gamma):
    golds, _ = ent2triplet(all_true_sub, all_true_ob, is_pred=False)
    preds, top_preds = ent2triplet(all_outs_sub, all_outs_ob, all_reps_sub, all_reps_ob, model, prompts_embedding,
                                   classes_to_id, gamma, is_pred=True)
    single_golds = []
    single_preds = []
    multi_golds = []
    multi_preds = []
    for g, p, tp in zip(golds, preds, top_preds):
        if len(g) == 1:
            single_golds.append(g)
            single_preds.append(tp)
        else:
            multi_golds.append(g)
            multi_preds.append(p)

    return single_preds, single_golds, multi_preds, multi_golds

