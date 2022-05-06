import numpy as np
from collections import defaultdict

from typing import List, Union, NamedTuple, Tuple, Counter
from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        #'''NOTE: You need to finish the code of computing f1-score.
        pred_true = 0
        pred_total = 0
        pred_tuple = extract_entities(predictions)
        label_tuple = extract_entities(labels)
        for p, l in zip(pred_tuple, label_tuple):
            pred_true += len(list(set(p).intersection(set(l))))
            pred_total += len(p) + len(l)

        f1 = 2 * pred_true / pred_total
        #'''
        return {"f1": f1}


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        # '''NOTE: You need to finish the code of computing f1-score.
        pred_true = 0
        pred_total = 0
        resized_predictions = predictions.transpose(2, 0, 1)

        pred1_tuple = extract_entities(resized_predictions[0], for_nested_ner=True, first_labels=True)
        pred2_tuple = extract_entities(resized_predictions[1], for_nested_ner=True, first_labels=False)
        labels1_tuple = extract_entities(labels1, for_nested_ner=True, first_labels=True)
        labels2_tuple = extract_entities(labels2, for_nested_ner=True, first_labels=False)

        pred_tuple = []
        for t1, t2 in zip(pred1_tuple, pred2_tuple):
            pred_tuple.append(list(set(t1).union(set(t2))))

        label_tuple = []
        for l1, l2 in zip(labels1_tuple, labels2_tuple):
            label_tuple.append(list(set(l1).union(set(l2))))

        for p, l in zip(pred_tuple, label_tuple):
            pred_true += len(list(set(p).intersection(set(l))))
            pred_total += len(p) + len(l)

        f1 = 2 * pred_true / pred_total
        # '''
        return {"f1": f1}


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    
    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.
    for idx in range(batch_labels_or_preds.shape[0]):
        batch_entities.append([])
        sentence = batch_labels_or_preds[idx]
        sentence_trans = []
        for bit in range(sentence.shape[0]):
            sentence_trans.append(id2label[sentence[bit]])
        cur_start = None
        cur_label = defaultdict(int)
        cur_end = None
        cur_state = 'seek_b'
        for i, l in enumerate(sentence_trans):
            if cur_state == 'seek_b' and l[0] == 'B':
                cur_start = i
                cur_label[l.split('-')[1]] += 1
                cur_end = i
                cur_state = 'seek_i'
            elif cur_state == 'seek_i':
                if l[0] == 'I':
                    cur_label[l.split('-')[1]] += 1
                    cur_end = i
                else:
                    cur_entity_cnt = list(sorted(zip(cur_label.values(), cur_label.keys())))
                    if len(cur_entity_cnt) == 1:
                        batch_entities[idx].append((cur_start, cur_end, cur_entity_cnt[0][1]))
                    else:
                        if cur_entity_cnt[-1][0] > cur_entity_cnt[-2][0]:
                            batch_entities[idx].append((cur_start, cur_end, cur_entity_cnt[-1][1]))
                        else:
                            max_cnt = cur_entity_cnt[-1][0]
                            max_label = []
                            for lab in cur_label.keys():
                                if cur_label[lab] == max_cnt:
                                    max_label.append(lab)
                            for lab in reversed(list(_LABEL_RANK.keys())):
                                if lab in max_label:
                                    ll = lab
                                    break
                            batch_entities[idx].append((cur_start, cur_end, ll))
                    if l[0] != 'B':
                        cur_start = None
                        cur_label = defaultdict(int)
                        cur_end = None
                        cur_state = 'seek_b'
                    else:
                        cur_start = i
                        cur_label = defaultdict(int)
                        cur_label[l.split('-')[1]] += 1
                        cur_end = i
                        cur_state = 'seek_i'
    # '''
    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')