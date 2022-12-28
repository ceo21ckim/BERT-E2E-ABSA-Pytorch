import argparse
from settings import * 

parser = argparse.ArgumentParser(description='setting parameter for training and evaluating..')

parser.add_argument('--semeval_dir', type=str, default=SEM_DIR, help='SemEval dataset directory path.')
parser.add_argument('--absa_type', type=str, default='san', help='self attention networks.')
parser.add_argument('--model_type', type=str, default='bert')
parser.add_argument('--fix_tfm', type=int, default=0, help='0 is not fixed parameters, 1 is fixed parameters, i.e., frozen model weight.')
parser.add_argument('--max_seq_length', type=int, default=512, help='Max sequence length of users reviews in each restaurant.')
parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs for training.')
parser.add_argument('--batch_size', type=int, default=128, help='The number of batch size for training and evaluating.')
parser.add_argument('--save_steps', type=int, default=100, help='checkpoint during training.')
parser.add_argument('--seed', type=int, default=42, help='setting seed.')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon is parameter in AdamW optimizer.')
parser.add_argument('--warmup_steps', type=int, default=0, help='if you want to use warmup step before training step. Using warmup_steps that is not zero.')
parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')

args = parser.parse_args()