import argparse
exam_code = '''
e.g)  
python evaluate.py -d ./models
'''
parser = argparse.ArgumentParser("Evaluate models",epilog=exam_code)   


parser.add_argument('-d'  ,'--directory'  ,default='./models' ,metavar='{...}'    ,help='directory path containing the models')
parser.add_argument('-p'  ,'--path'       ,default='./models/ae0_80.pt'       , help='Specify the model path')
parser.add_argument('-th', '--threshold'  ,default=0.004,type=float , help='Threshold of AE')
parser.add_argument('--dataset_path'      ,default='./data/split/test.csv'  , help='test dataset path')
parser.add_argument('-s','--save'         ,default=True, type=bool  , help='whether to save')


args = parser.parse_args()

