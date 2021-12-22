from torch import nn
def get_loss(name:str='crossentropy',task_type='cls'):
    loss = None
    name = name.lower()
    if name.endswith('loss'):
        name = name.replace('loss','')
        
    if task_type == 'cls':
        if name == 'crossentropy':
            loss = nn.CrossEntropyLoss()
        elif name == 'multimargin':
            loss = nn.MultiMarginLoss()
        elif name == 'nll':
            loss = nn.NLLLoss()
        else:
            ValueError(f'there is No {name} loss!!!')
    
    elif task_type == 'reg':
        if name == 'mse':
            loss = nn.MSELoss()
        elif name == 'bce':
            loss = nn.BCELoss()
        elif name == 'bcewithlogits':
            loss = nn.BCEWithLogitsLoss()
        elif name == 'hingeembedding':
            loss = nn.HingeEmbeddingLoss()
        elif name == 'huber':
            loss = nn.HuberLoss()
        elif name == 'kldiv':
            loss = nn.KLDivLoss()
        elif name == 'l1':
            loss = nn.L1Loss()
        elif name == 'multilabelsoftmargin':
            loss = nn.MultiLabelSoftMarginLoss()
        elif name == 'poissonnll':
            loss = nn.PoissonNLLLoss()
        elif name == 'smoothl1':
            loss = nn.SmoothL1Loss()
        elif name == 'softmargin':
            loss = nn.SoftMarginLoss()
            
        else:
            ValueError(f'there is No {name} loss!!!')
    return loss