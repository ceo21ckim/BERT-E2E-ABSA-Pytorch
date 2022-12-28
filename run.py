import pickle

from transformers import BertTokenizer, BertConfig
import torch 

from utils import * 
from parsers import args 

from models.absa_bert import *

output_dir = f'{args.model_type}-{args.absa_type}'
args.output_dir = output_dir


if __name__ == '__main__':

    processor = ABSAProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = BertConfig, BertABSATagger, BertTokenizer
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    config.absa_type = args.absa_type
    config.fix_tfm = args.fix_tfm

    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.load_state_dict(torch.load(args.param_dir))
    model.to(args.device)

    test_dataloader = get_absa_loader(args, tokenizer, mode='test')

    results = absa_evaluate(args, model, test_dataloader, mode='test')

    save_dir = os.path.join(RESULT_DIR, 'inference_results.pkl')
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    with open(save_dir, 'wb') as f:
        pickle.dump(results, f)