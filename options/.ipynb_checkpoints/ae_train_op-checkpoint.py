import argparse
import torch
exam_code = '''
e.g)  
python -m ae.train 
'''
parser = argparse.ArgumentParser("Train datasets",epilog=exam_code)   

# parser.add_argument('-d'  ,'--dt'      ,default='pf'      ,metavar='{pf,bln}' , help='Dataset')
parser.add_argument('--device'   ,default=None,type=str     ,help='cpu | gpu')

parser.add_argument('-m'  ,'--model'   ,default='ae0' ,metavar='{...}'    ,help='model name')
parser.add_argument('--batch_size'   ,default=None,type=int     ,help='batch size')

parser.add_argument('--lr'     ,default=None,type=float     ,help='Learning Rate')
parser.add_argument('--loss'   ,default=None,type=str     ,help='which loss crossentropy(default)|multimargin|nll ?')

parser.add_argument('-f'       ,help='for ipynb')
args = parser.parse_args()

args.model = args.model.lower()
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'