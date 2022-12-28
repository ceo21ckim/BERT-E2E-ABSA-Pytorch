from tqdm import tqdm 
from settings import * 

import pandas as pd 
import itertools
from collections import OrderedDict
import torch 
import os
import pickle 
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import get_linear_schedule_with_warmup

SMALL_POSITIVE_CONST = 1e-4

def get_absa_loader(args, tokenizer, mode):
    '''
    Input:
        args: configuration
        tokenizer: tokenizer (Bert-Tokenizer)
    
    Returns:
        dataset: dataset consist of input_ids, input_mask, segment_ids, label_ids according to dataset
        all_evaluate_label_ids: label_ids (such as E-POS, I-POS, S-NEG, etc..) 
        
    examples:
        (tensor([ 101, 2021, 1996, 3095, 2001, 2061, 9202, 2000, 2149, 1012,  102,    0,
                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0]),

         tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),

         tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),

         tensor([0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    '''
    
    processor = ABSAProcessor()

    # cache' name
    cached_features_file = os.path.join(args.semeval_dir, 
                                        'cached_{}_{}_{}'.format(
                                        mode,
                                        list(filter(None, args.model_name_or_path.split('/'))).pop(), 
                                        str(args.max_seq_length)))


    if os.path.exists(cached_features_file):
        print('cached_features_file:', cached_features_file)
        features = torch.load(cached_features_file)

    else:
        label_list = processor.get_labels()
        examples = processor.get_examples(args.semeval_dir, mode)

        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_segment_id=0, pad_token_segment_id=0)

        torch.save(features, cached_features_file)


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if mode=='train':
        sampler = RandomSampler(dataset)
    elif mode=='valid' or mode=='test':
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    
    return (dataloader, all_evaluate_label_ids)



def absa_evaluate(args, model, dataloader, mode='valid'):

    results = {}

    eval_loss, eval_steps = 0.0, 0
    preds = None
    out_label_ids = None
    
    model.eval()
    for batch in tqdm(dataloader[0], desc='Evaluating...'):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0], 
                      'attention_mask': batch[1], 
                      'token_type_ids': batch[2], 
                      'labels':         batch[3]}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        
    eval_loss /= eval_steps 

    preds = np.argmax(preds, axis=-1)
    result = compute_metrics_absa(preds, out_label_ids, dataloader[1])
    result['eval_loss'] = eval_loss 
    results.update(result)

    output_file = os.path.join(args.output_dir, '%s_results.txt' % mode)

    with open(output_file, 'w')  as writer:
        for key in sorted(result.keys()):
            if 'eval_loss' in key:
                print(f'{key} = {str(result[key])}')
            
            writer.write('%s = %s\n' % (key, str(result[key])))

    return results


def absa_train(args, model, train_loader, eval_loader, optimizer):

    t_total = len(train_loader) * args.num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    best_loss = float(np.inf)
    global_step = 0
    train_loss_list, eval_loss_list = [], []
    results_list = []
    set_seed(args)
    print('***** Running Training *****')
    for epoch in range(1, args.num_epochs + 1):
        train_loss, eval_loss = 0.0, 0.0
        train_iterator = tqdm(train_loader, desc='training...', disable=False)
        model.train()
        
        for batch in train_iterator:
            
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':          batch[0], 
                      'attention_mask':     batch[1],
                      'token_type_ids':     batch[2], 
                      'labels':             batch[3]}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

            train_loss += loss.item() / len(batch)
            optimizer.step()
            scheduler.step()
            global_step += 1


            # model check-point
            if global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model_to_save = model.module if hasattr(model, 'module') else model 
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            

            train_loss /= len(train_iterator)
            train_loss_list.append(train_loss)



        model.eval()
        eval_iterator = tqdm(eval_loader[0], desc='training...')
        preds, out_label_ids = None, None
        results = {}
        for batch in eval_iterator:
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':          batch[0], 
                        'attention_mask':     batch[1],
                        'token_type_ids':     batch[2], 
                        'labels':             batch[3]}
                
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss /= len(eval_iterator)

        eval_loss_list.append(eval_loss)
        preds = np.argmax(preds, axis=-1)
        result = compute_metrics_absa(preds, out_label_ids, eval_loader[1])
        results.update(result)
        results_list.append(results)

        output_file = os.path.join(args.output_dir, '%s_results.txt' % 'valid')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(output_file, 'w') as writer:
            for key in sorted(result.keys()):
                if 'eval_loss' in key:
                    print(f'{key} = {str(result[key])}')
                writer.write('%s = %s\n' % (key, str(result[key])))

        if best_loss > eval_loss :
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best-parameters.pt'))
            best_loss = eval_loss
            best_epoch = epoch

        print(f'epochs: [{epoch}/{args.num_epochs}], best epoch is {best_epoch}')
        print(f'train loss: {train_loss:.5f},\t valid loss: {eval_loss:.5f}')
        print(f'macro-f1: {result["macro-f1"]:.5f},\tmicro-f1: {result["micro-f1"]:.5f}')
        print(f"recall: {result['recall']:.5f},\tprecision: {result['precision']:.5f}")
    print('***** Finished Training *****')

    return {
        'results': results_list, 
        'train_loss': train_loss_list, 
        'valid_loss': eval_loss_list
    }


'''
SimEval B-POS(NEG), I-POS(NEG), E-POS(NEG), O-POS(NEG), S-POS(NEG), EQ-POS(NEG) 형태로 입력됩니다.
O-POS(NEG), EQ-POS(NEG)는 아무 속성이 없는 단어를 의미하고,
B-POS(NEG), I-POS(NEG), E-POS(NEG), S-POS(NEG)는 단어의 속성을 나타냅니다.

original paper: https://arxiv.org/pdf/1910.00883.pdf
'''
def ot2bieos_ts(ts_tag_sequence):
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        
        else:
            cur_pos, cur_sentiment = cur_ts_tag = cur_ts_tag.split('-')
            # cur_pos is 'T'
            if cur_pos != prev_pos:
                
                if i == n_tags -1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
                    
            else:
                if i == n_tags -1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos 
    return new_ts_sequence

def ot2bieos_ts_batch(ts_tag_seqs):
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = ot2bieos_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def tag2ts(ts_tag_sequence):
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    begin, end = -1, -1
    
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        eles = ts_tag.split('-')
        
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        
        if sentiment != 'O':
            sentiments.append(sentiment)
        
        if pos == 'S':
            ts_sequence.append((i, i, sentiment))
            sentiments = []
            
        elif pos == 'B':
            beg = i 
            if len(sentiments) > 1:
                sentiments = [sentiments[-1]]
        
        elif pos == 'E':
            end = i
            
            if end > begin > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((begin, end, sentiment))
                sentiments = []
                begin, end = -1, -1
                
    return ts_sequence


def logsumexp(tensor, dim=-1, keepdim=True):
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim).log())

class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        '''
        Inputs:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single 
                    sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            label: (Optional) string. The label of the example. This should be specified for train and dev examples.
        '''
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class SeqInputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, evaluate_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.evaluate_label_ids = evaluate_label_ids

class InferenceProcessor:
    def get_examples(self, dataframe):
        return self._create_examples(dataframe)
    
    def get_labels(self):
        return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS', 
                 'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG', 
                 'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']

    def _create_examples(self, dataframe):
        examples = []

        for sample_id, sentence in enumerate(dataframe.loc[:, 'text'].values):
            guid = f'inference-{sample_id}'
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, label=None))
        
        return examples 



class ABSAProcessor:
    def get_examples(self, data_dir, set_type='train'):
        return self._create_examples(data_dir=data_dir, set_type=set_type)
    
    def get_labels(self):
        return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS', 
                 'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG', 
                 'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
    
    def _create_examples(self, data_dir, set_type):
        examples = []
        file = os.path.join(data_dir, '%s.txt' % set_type)
        class_count = np.zeros(3)
        with open(file, 'r', encoding='UTF-8') as fp:
            sample_id = 0 
            for line in fp:
                '''
                sent_string: But the staff was so horrible to us.
                tag_string: But=O the=O staff=T-NEG was=O so=O horrible=O to=O us=O .=O

                words: [But, the, staff, was, so, horrible, to, us, .]
                tags: [O, O, S-NEG, O, O, O, O, O, O]
                '''
                sent_string, tag_string = line.strip().split('####')

                words, tags =[], []
                for tag_item in tag_string.split(' '):
                    eles = tag_item.split('=')
                    if len(eles) == 1:
                        raise Exception('Invalid samples %s...' % tag_string)

                    elif len(eles) == 2:
                        word, tag = eles

                    else:
                        word = ''.join((len(eles) - 2) * ['='])
                        tag = eles[-1]

                    words.append(word)
                    tags.append(tag)

                tags = ot2bieos_ts(tags)

                guid = '%s-%s' % (set_type, sample_id)
                text_a = ' '.join(words)
                gold_ts = tag2ts(ts_tag_sequence=tags)
                for (_, _, s) in gold_ts:
                    if s == 'POS':
                        class_count[0] += 1

                    if s == 'NEG':
                        class_count[1] += 1

                    if s == 'NEU':
                        class_count[2] += 1

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags))
                sample_id += 1

            
            print('%s class count: %s' % (set_type, class_count))
            print(f'review length: {sample_id:,}\t total class count: {int(sum(class_count)):,}')
            return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break 
        
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        
        else:
            tokens_b.pop()

            
            
def convert_examples_to_seq_features(examples, label_list, tokenizer, 
                                     cls_token='[CLS]', sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0, 
                                     cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True):
    
    label_map = {label:i for i, label in enumerate(label_list)}
    features = []
    max_seq_length = -1 
    examples_tokenized = []
    for example in examples:
        tokens_a, labels_a = [], []
        evaluate_label_ids = []
        words = example.text_a.split(' ')
        wid, tid = 0, 0
        for word, label in zip(words, example.label):
            subwords = tokenizer.tokenize(word)
            tokens_a.extend(subwords)
            if label != 'O':
                labels_a.extend([label] + ['EQ'] * (len(subwords) - 1))
            else:
                labels_a.extend(['O'] * len(subwords))
            evaluate_label_ids.append(tid)
            wid += 1
            tid += len(subwords)
        
        assert tid == len(tokens_a)
        
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        examples_tokenized.append((tokens_a, labels_a, evaluate_label_ids))
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    
    max_seq_length += 2
    for (tokens_a, labels_a, evaluate_label_ids) in examples_tokenized:
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        labels = labels_a + ['O']
        
        # if cls_token_at_end
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        labels = ['O'] + labels
        evaluate_label_ids += 1
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        label_ids = [label_map[label] for label in labels]
        
        # if pad_on_left
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        
        label_ids = label_ids + ([0] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        features.append(SeqInputFeatures(input_ids=input_ids, 
                                         input_mask=input_mask,
                                         segment_ids=segment_ids,
                                         label_ids=label_ids,
                                         evaluate_label_ids=evaluate_label_ids))
    print('maximal sequence length is %d' % (max_seq_length))
    return features 

    
    
def convert_examples_to_features(examples, label_list, max_seq_length, 
                                 tokenizer, cls_token='[CLS]', sep_token='[SEP]', 
                                 pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1, 
                                 cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True):
    
    label_map = {label:i for i, label in enumerate(label_list)}
    
    features = []
    
    for example in examples:

        tokens_a = tokenizer.tokenize(example.text_a)
        
        tokens_b = None
        
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3) # [CLS], [SEP], [SEP]
        
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        '''
        tokens: [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        type_ids: 0    0   0    0    0      0    0   0    1  1  1  1  1   1
        '''
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)
        
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = label_map[example.label]
            
        features.append(InputFeatures(input_ids=input_ids, 
                                     input_mask=input_mask,
                                     segment_ids=segment_ids, 
                                     label_id=label_id))
        
    return features 



def match_ts(gold_ts_sequence, pred_ts_sequence):
    '''
    Inputs:
        gold_ts_sequence: gold standard targeted sentiment sequence (ground truth)
        pred_ts_sequence: predicted targeted sentiment sequence
    '''
    
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
        
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count



def compute_metrics_absa(preds, labels, all_evaluate_label_ids):
    absa_label_vocab = {'O':0, 'EQ':1, 'B-POS':2, 'I-POS':3, 'E-POS':4, 'S-POS':5,
                        'B-NEG':6, 'I-NEG':7, 'E-NEG':8, 'S-NEG':9, 'B-NEU':10, 
                        'I-NEU':11, 'E-NEU':12, 'S-NEU':13}
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k 
        
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    n_samples = len(all_evaluate_label_ids)

    class_count = np.zeros(3)
    
    for i in range(n_samples):
        evaluate_label_ids = all_evaluate_label_ids[i]
        pred_labels = preds[i][evaluate_label_ids]
        gold_labels = labels[i][evaluate_label_ids]
        assert len(pred_labels) == len(gold_labels)
        
        pred_tags = [absa_id2tag[label] for label in pred_labels]
        gold_tags = [absa_id2tag[label] for label in gold_labels]
        
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)
        
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)
        
        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        
        for (_, _, s) in g_ts_sequence:
            if s == 'POS':
                class_count[0] += 1
            if s == 'NEG':
                class_count[1] += 1
            if s == 'NEU':
                class_count[2] += 1
    
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts +SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)
        
    macro_f1 = ts_f1.mean()
    
    n_tp_total = sum(n_tp_ts)
    n_g_total = sum(n_gold_ts)
    print('class_count:', class_count)
    
    n_p_total = sum(n_pred_ts)
    micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + SMALL_POSITIVE_CONST)
    scores = {'macro-f1':macro_f1, 'precision':micro_p, 'recall':micro_r, 'micro-f1':micro_f1}

    return scores 


###########################################################
def get_pos_idx(x):
    if (x > 1) and (x < 6):
        return True 
    else: return False

def get_neg_idx(x):
    if (x > 5) and (x < 10):
        return True 
    else:
        return False 

def get_opinion_aspect(texts, tags,  tokenizer):
    texts = torch.concat(texts)
    tags = torch.concat(tags)

    pos_reviews, neg_reviews = [], []
    for txt, tag in tqdm(zip(texts, tags), desc=f'extract opinion in reviews, total iteration: {len(tags):,}'):
        pos_idx = list(map(get_pos_idx, tag))
        neg_idx = list(map(get_neg_idx, tag))
        pos_reviews.append(tokenizer.convert_ids_to_tokens(txt[pos_idx]))
        neg_reviews.append(tokenizer.convert_ids_to_tokens(txt[neg_idx]))
    
    return pos_reviews, neg_reviews


def word_similarity(model, word_vocabs, opinion):
    # Food, Service, Ambiance, Price, and Location

    results = OrderedDict()
    for word in word_vocabs:
        word = word.split()
        for w in word:
            if w:
                try:
                    results[w] = model.similarity(opinion, w)
                except:
                    pass 
    
    return results 


def sentiment_filtering(tag_words, main_opinion):
    results = []
    for sent in tag_words:
        word_aspect = []
        
        for aspect in main_opinion:
            scores = []
            
            for word in sent:
                
                try:
                    scores.append(aspect[word])
                except:
                    pass 
            
            word_aspect.append(np.mean(scores) if scores != [] else 0 )

        results.append(word_aspect)
    return np.array(results)


def get_tags(dataframe, tag_words, main_opinion):
    # Food, Service, Ambiance, Price, and Location
    df = dataframe.copy()
    outputs = sentiment_filtering(tag_words, main_opinion)

    df.loc[:, ['food', 'service', 'ambiance', 'price', 'location']] = outputs

    user_group = df.loc[:, ['user_id', 'food', 'service', 'ambiance', 'price', 'location']].groupby('user_id').mean()
    item_group = df.loc[:, ['business_id', 'food', 'service', 'ambiance', 'price', 'location']].groupby('business_id').mean()

    users = []
    for idx, row in user_group.iterrows():
        users.append([idx, row.idxmax()])

    items = []
    for idx, row in item_group.iterrows():
        items.append([idx, row.idxmax()])

    user_tags = pd.DataFrame(users, columns=['user_id', 'user_tag'])
    item_tags = pd.DataFrame(items, columns=['business_id', 'rest_tag'])
    return user_tags, item_tags 




def inference(args, loader, model, tokenizer):
    texts, tags = [], []
    
    if not os.path.exists('reviews.pkl'):
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc='Evaluating...'):
                input_ids = batch[0]

                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[0], 
                        'attention_mask': batch[1], 
                        'token_type_ids': batch[2]}

                outputs = model(**inputs)[0]
                outputs = torch.argmax(outputs, dim=-1)

                texts.append(input_ids)
                tags.append(outputs.detach().cpu())

        with open('reviews.pkl', 'wb') as f:
            pickle.dump(texts, f)

        with open('taggings.pkl', 'wb') as f:
            pickle.dump(tags, f)
    
    else:
        with open('reviews.pkl', 'rb') as f:
            texts = pickle.load(f)
        
        with open('taggings.pkl', 'rb') as f:
            tags = pickle.load(f)

    pos_words, neg_words = get_opinion_aspect(texts, tags, tokenizer)

    pos_vocabs = list(itertools.chain(*pos_words))
    neg_vocabs = list(itertools.chain(*neg_words))

    return pos_words, neg_words, pos_vocabs, neg_vocabs 



