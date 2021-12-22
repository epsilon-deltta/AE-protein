import torch
def train(dl,model,lossf,opt,device='cuda'):
    model.train()
    for x,y in dl:
        x,y = x.to(device),y.to(device)
        pre = model(x)
        loss = lossf(pre,y)

        opt.zero_grad()
        loss.backward()
        opt.step()

def test(dl,model,lossf,epoch=None,exist_acc=True,device='cuda'):
    model.eval()
    size, acc , losses = len(dl.dataset) ,0,0
    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(device),y.to(device)
            pre = model(x)
            loss = lossf(pre,y)
            
            if exist_acc: 
                acc += (pre.argmax(1)==y).type(torch.float).sum().item()
            losses += loss.item()
    if exist_acc:
        accuracy = round(acc/size,4)
    else:
        accuracy = None
    val_loss = round(losses/size,6)
    print(f'[{epoch}] acc/loss: {accuracy}/{val_loss}')
    return accuracy,val_loss

import copy
def run(trdl,valdl,model,loss,opt,epoch=100,patience = 5,exist_acc=True,device='cuda'):
    val_losses = {0:1}
    model = model.to(device)
    for i in range(epoch):
        train(trdl,model,loss,opt,device=device)
        acc,val_loss = test(valdl,model,loss,epoch=i,exist_acc=exist_acc,device=device)


        if min(val_losses.values() ) > val_loss:
            val_losses[i] = val_loss
            best_model = copy.deepcopy(model)
        if i == min(val_losses,key=val_losses.get)+patience:
            break
    return best_model,val_losses

if __name__ == '__main__':

    from options.ae_train_op import args
    from models import *
    
    # model load
    model,config = get_model(args.model)
    
    # re-config
    transform  = config['transform'] 
    batch_size = config['batch_size'] if args.batch_size is None else args.batch_size
    loss       = config['loss'] if args.loss is None else args.loss
    lr         = 0.001 if args.lr is None else args.lr
    
    # settings
    from utils import get_loss
    loss = get_loss(name=loss,task_type='reg')
    params = [p for p in model.parameters() if p.requires_grad]
    opt  = torch.optim.Adam(params,lr=lr)

    # dataset and loader
    
    from dataset import ProteinDataset

    trdt  = ProteinDataset('./data/split/train.csv',transform=transform)
    valdt = ProteinDataset('./data/split/val.csv'  ,transform=transform)
    # tedt  = ProteinDataset('./data/split/test.csv' ,transform=transform)

    trdl  = torch.utils.data.DataLoader(trdt, batch_size=batch_size, num_workers=4)
    valdl  = torch.utils.data.DataLoader(valdt, batch_size=batch_size, num_workers=4)
    # tedl  = torch.utils.data.DataLoader(tedt, batch_size=batch_size, num_workers=4)


    # train/validate
    best_model,val_losses = run(trdl,valdl,model,loss,opt,device=args.device,exist_acc=config['exist_acc'])

    # save
    import os
    model_name = f"{best_model.__str__().split('(')[0]}_{max(val_losses)}.pt"
    model_path = os.path.join('./models',model_name) 
    torch.save(best_model.state_dict(),model_path)