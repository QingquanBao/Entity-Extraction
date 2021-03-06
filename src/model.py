
from audioop import bias
from typing import Optional
from unicodedata import bidirectional

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.file_utils import ModelOutput

from ee_data import EE_label2id1, NER_PAD

NER_PAD_ID = EE_label2id1[NER_PAD]


@dataclass
class NEROutputs(ModelOutput):
    """
    NOTE: `logits` here is the CRF decoding result.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.LongTensor] = None


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.num_labels = num_labels
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        self.loss_fct = CrossEntropyLoss()

    def _pred_labels(self, _logits):
        return torch.argmax(_logits, dim=-1)

    def forward(self, hidden_states, labels=None, no_decode=False, only_logits=False):
        _logits = self.layers(hidden_states)
        if only_logits: return _logits
        loss, pred_labels = None, None

        if labels is None:
            # Test dataset 
            pred_labels = self._pred_labels(_logits)    
        else:
            loss = self.loss_fct(_logits.view(-1, self.num_labels), labels.view(-1))
            if not no_decode:
                # Validation set
                pred_labels = self._pred_labels(_logits)

        return NEROutputs(loss, pred_labels)


class CRFClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.num_labels = num_labels

        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

        self.crf    = CRF(num_labels, batch_first=True)

    def _pred_labels(self, emissions, mask, padding_value):

        pred_labels = self.crf.decode(emissions, mask.byte())
        out_ = pad_sequence([torch.tensor(pred_label, device=emissions.device) for pred_label in pred_labels],
                             batch_first=True, padding_value=padding_value)
        return out_

    def forward(self, hidden_states, attention_mask, labels=None, no_decode=False, label_pad_token_id=NER_PAD_ID, only_logits=False):    
      
        _emissions = self.layers(hidden_states)
        if only_logits: return _emissions
        loss, pred_labels = None, None

        if labels is None:
            # Test dataset 
            pred_labels = self._pred_labels(_emissions, attention_mask, padding_value=label_pad_token_id)    
        else:
            loss = -1 * self.crf(_emissions, labels, mask=attention_mask.byte())
            if not no_decode:
                # Validation set
                pred_labels = self._pred_labels(_emissions, attention_mask, padding_value=label_pad_token_id)

        return NEROutputs(loss, pred_labels)


def _group_ner_outputs(output1: NEROutputs, output2: NEROutputs):
    """ logits: [batch_size, seq_len] ==> [batch_size, seq_len, 2] """
    grouped_loss, grouped_logits = None, None

    if not (output1.loss is None or output2.loss is None):
        grouped_loss = (output1.loss + output2.loss) / 2

    if not (output1.logits is None or output2.logits is None):
        grouped_logits = torch.stack([output1.logits, output2.logits], dim=-1)

    return NEROutputs(grouped_loss, grouped_logits)

class BertBasedModel(BertPreTrainedModel):
    config_class = BertConfig
    def __init__(self, config: BertConfig):
        super().__init__(config) 
        self.config = config
        self.bert = BertModel(config)

    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        return embedding_output

    def encode(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        inputs_embeds=None,
    ):
        outputs = self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
        last_hidden_state = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states  # num_layers + 1 (embeddings)
        return last_hidden_state, all_hidden_states
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
            fwd_type=0,
            embed=None,
    ):
        if fwd_type == 0:
            # Normal outputs
            last_hidden_state, all_hidden_states = self.encode(
                input_ids, token_type_ids, attention_mask
            )
            output = self.hidden2out(last_hidden_state, attention_mask=attention_mask, labels=labels, labels2=labels2, no_decode=no_decode)
        elif fwd_type == 1:
            # Only return Input Embeddings
            return self.embed_encode(input_ids, token_type_ids, attention_mask)
        elif fwd_type == 2:
            # Forward w/ Input Embeddings
            assert embed is not None
            last_hidden_state, all_hidden_states = self.encode(
                None, token_type_ids, attention_mask, embed
            )

            output = self.hidden2out(last_hidden_state, attention_mask=attention_mask, labels=labels, labels2=labels2, no_decode=no_decode, only_logits=True)

        return output 

    def hidden2out(
                    self, 
                    hiddens, 
                    labels,
                    labels2=None, 
                    no_decode=False, 
                    only_logits=False):
        pass

class BertForLinearHeadNERv2(BertBasedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def hidden2out(self, 
                    hidden,
                    attention_mask,
                    labels, 
                    labels2=None, 
                    no_decode=False, 
                    only_logits=False):
        return self.classifier(hidden, labels, no_decode=no_decode, only_logits=only_logits)

class BertForCRFHeadNestedNERv2(BertBasedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier1 = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)

        self.init_weights()
    
    def hidden2out(
                    self, 
                    hiddens, 
                    attention_mask,
                    labels=None,
                    labels2=None, 
                    no_decode=False, 
                    only_logits=False):
        out1 = self.classifier1(hiddens, attention_mask, labels, no_decode=no_decode, only_logits=only_logits)
        out2 = self.classifier2(hiddens, attention_mask, labels2, no_decode=no_decode, only_logits=only_logits)
        if only_logits:
            return (out1, out2)
        else:
            return _group_ner_outputs(out1, out2)

class BertForLinearHeadNestedNERv2(BertBasedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier1 = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = LinearClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)

        self.init_weights()

    def hidden2out(
                    self, 
                    hiddens, 
                    attention_mask,
                    labels=None,
                    labels2=None, 
                    no_decode=False, 
                    only_logits=False):
        out1 = self.classifier1(hiddens,  labels, no_decode=no_decode, only_logits=only_logits)
        out2 = self.classifier2(hiddens,  labels2, no_decode=no_decode, only_logits=only_logits)
        if only_logits:
            return out1, out2
        else:
            return _group_ner_outputs(out1, out2)

class BertForLinearHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, labels, no_decode=no_decode)
        return output

class BertForLinearHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier1 = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = LinearClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        '''NOTE: This is where to modify for Nested NER.

        Use the above function _group_ner_outputs for combining results.

        '''
        output1 = self.classifier1.forward(sequence_output, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, labels2, no_decode=no_decode)
        return _group_ner_outputs(output1, output2)


class BertForCRFHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.classifier = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        
        return output

class BertForCRFHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier1 = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        '''NOTE: This is where to modify for Nested NER.

        Use the above function _group_ner_outputs for combining results.

        '''
        output1 = self.classifier1.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, attention_mask, labels2, no_decode=no_decode)
        return _group_ner_outputs(output1, output2)