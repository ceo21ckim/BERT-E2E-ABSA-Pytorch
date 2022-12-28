import pickle

from utils import * 
from parsers import args
from absa_bert import BertABSATagger

from transformers import BertTokenizer, BertConfig
from torch import optim


if __name__ == '__main__':
    output_dir = f'{args.model_type}-{args.absa_type}'
    args.output_dir = output_dir

    processor = ABSAProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = BertConfig, BertABSATagger, BertTokenizer
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    config.absa_type = args.absa_type
    config.fix_tfm = args.fix_tfm

    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model = model.to(args.device)

    train_dataloader, _ = get_absa_loader(args, tokenizer, mode='train')
    valid_dataloader = get_absa_loader(args, tokenizer, mode='valid')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    results = absa_train(args, model, train_dataloader, valid_dataloader, optimizer)

    save_dir = os.path.join(RESULT_DIR, 'valid_results.pkl')
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    with open(save_dir, 'wb') as f:
        pickle.dump(results, f)
    
    '''
    if you want load pickle file(results), excute code follows as:
    
    with open(file_name, "rb") as f:
        dataset = pickle.load( f)
    
    Returns (example):
        {'results': [{'macro-f1': 0.3989729576446897,
        'precision': 0.6348683166335006,
        'recall': 0.6666665515256388,
        'micro-f1': 0.6503290310922997},
        {'macro-f1': 0.4704771809728931,
        'precision': 0.663250268923826,
        'recall': 0.7823832845624725,
        'micro-f1': 0.7178583116318107},
        {'macro-f1': 0.49240466168529506,
        'precision': 0.6812864501043202,
        'recall': 0.8048357850024551,
        'micro-f1': 0.7378758060928258}],
        'train_loss': [0.10271387666146806, 0.04328422926049283, 0.032776613858468986],
        'valid_loss': [0.17818615904876164, 0.13069180931363786, 0.11900808449302401]}
    '''