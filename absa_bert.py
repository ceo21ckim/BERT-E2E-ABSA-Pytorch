# coding=utf-8
# Copyright 2018 Google AI Language, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PreTrainedModel, BertModel, BertConfig
from torch.utils.data import Dataset

import torch 
from transformers import BertModel 
from torch import nn 

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

###########################################################################################################################

class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768

class SAN(nn.Module):
    def __init__(self, model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.model = model 
        self.self_attn = nn.MultiheadAttention(model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(model)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask) # (key, query, value)
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src 

class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BertABSATagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels
        self.tagger_config = TaggerConfig()
        self.bert = BertModel(bert_config)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        if bert_config.fix_tfm:
            for p in self.bert.parameters():
                p.required_grad = False  # Frizen
        
        self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
        self.tagger = SAN(model=bert_config.hidden_size, nhead=12, dropout=0.1)
        penultimate_hidden_size = self.tagger_config.hidden_size
        self.classifier = nn.Linear(penultimate_hidden_size, self.num_labels)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids, 
            position_ids=position_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask, 
            head_mask=head_mask
        )
        
        tagger_input = outputs[0] # pooler 
        tagger_input = self.bert_dropout(tagger_input)
        tagger_input = tagger_input.transpose(0, 1) # 각 성분에 대해서 classification을 하기 위함.
        classifier_input = self.tagger(tagger_input)
        classifier_input = classifier_input.transpose(0, 1)
        classifier_input = self.tagger_dropout(classifier_input)
        logits = self.classifier(classifier_input)
        
        outputs = (logits, ) + outputs[2:]
        
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = criterion(active_logits, active_labels)
            
            else:
                loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
                
            outputs = (loss, ) + outputs 
        
        return outputs 


class ABSADataset(Dataset):
    def __init__(self, args, dataframe, tokenizer):
        self.tokenizer = tokenizer 
        self.data = dataframe 
        self.reviews = dataframe.text 
        self.labels = dataframe.stars
        self.max_seq_length = args.max_seq_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        review = self.reviews[idx]

        inputs = self.tokenizer.encode_plus(
            review, 
            None,
            add_special_tokens=True, 
            max_length=self.max_seq_length, 
            padding='max_length', 
            return_token_type_ids=True, 
            truncation=True
        )

        input_ids = inputs['input_ids']
        masks = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return (
            torch.tensor(input_ids, dtype=torch.long), # token_ids
            torch.tensor(masks, dtype=torch.long), # attention_mask
            torch.tensor(token_type_ids, dtype=torch.long), # token_type_ids
            torch.tensor(self.labels[idx], dtype = float) # labels
        )