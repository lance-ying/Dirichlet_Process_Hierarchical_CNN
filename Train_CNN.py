import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GroupKFold
# from sklearn.model_selection import KFold
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model import * 
from tqdm import tqdm
import argparse
import random

# os.chdir("/home/lancelcy/PRIORI")
def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():

    print("loading data")

    os.mkdir("./checkpt")
    X=np.load("./IEMOCAP.npy")

    print(X.shape)

    df=pd.read_csv("./data.csv")

    Y=df["valence"].to_numpy()
    print(Y.shape)
    folds=len(df["sub_id"].unique())

    print("data loaded")
    kfold = GroupKFold(n_splits=folds)
    UAR=np.zeros((5,folds))


    count=0
    for train_idx, test_idx in tqdm(kfold.split(X,None, df["sub_id"])):
        path="./checkpt/fold_{}".format(count)
        if not os.path.exists(path):
            os.mkdir(path)
        np.random.shuffle(train_idx)
        Tr_idx=train_idx[:int(0.8*len(train_idx))]
        Val_idx=train_idx[int(0.8*len(train_idx)):]
        Tes_idx=test_idx
        X_train=np.take(X,Tr_idx,axis=0)
        X_val=np.take(X,Val_idx,axis=0)
        Y_train=np.take(Y,Tr_idx,axis=0)
        Y_val=np.take(Y,Val_idx,axis=0)
        X_test=np.take(X,Tes_idx,axis=0)
        Y_test=np.take(Y,Tes_idx,axis=0)
    
 
        for i in range(5):
            seed_everything(i+42)
            train_data=TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(Y_train))
            val_data=TensorDataset(torch.FloatTensor(X_val),torch.LongTensor(Y_val))
            train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader=DataLoader(val_data, batch_size=64, shuffle=True)
            
            model = CNN()

            best_model=None
            best_loss=np.inf
            epoch=0
            patience=5
            cur_patience=patience

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            if torch.cuda.is_available():
                model = model.cuda()
                criterion = criterion.cuda()

            #training
            while cur_patience!=0 and epoch<50:
                loss_train = 0
                loss_valid = 0
                model.train()
                it=iter(train_loader)
                for step in range(1,len(train_loader)+1):
                    mfbs, label = next(it)
                    mfbs=mfbs.cuda()
                    label=label.cuda()
                    optimizer.zero_grad()
                    prediction=model(mfbs)
                    loss=criterion(prediction, label)

                    loss.backward()
                    optimizer.step()
                    loss_train+=loss.item()
                
                new_train_loss=loss_train/len(train_loader)

                with torch.no_grad():
                    it=iter(val_loader)
                    for step in range(1,len(val_loader)+1):
                        mfbs, label = next(it)
                        mfbs=mfbs.cuda()
                        label=label.cuda()
                        prediction=model(mfbs)
                        loss=criterion(prediction, label)
                        loss_valid+=loss.item()
                new_val_loss=loss_valid/len(val_loader)
                print("epoch ", epoch, "train_loss=",new_train_loss,"val_loss=",new_val_loss)

                if new_val_loss<=best_loss:
                    best_model=model.state_dict()
                    best_loss=new_val_loss
                    cur_patience=patience
                else:
                    cur_patience-=1

                epoch+=1
            with torch.no_grad():
                model= CNN()
                model.load_state_dict(best_model)
                Y_out=model(torch.from_numpy(X_test)).numpy()
            Y_pred=np.zeros(Y_test.shape[0])
            for j, pred in enumerate(Y_out):
                Y_pred[j]=np.argmax(pred)
            perf=metrics.recall_score(Y_test,Y_pred,average="macro")
            print(count, i, perf)
            UAR[i,count]=perf
            # print(i)
            filename=os.path.join(path,"{}.checkpoint".format(i))
            # print(filename)
            with torch.no_grad():
                torch.save(best_model, filename)
        count+=1
    
    np.save("./IEMOCAP_cnn.npy",UAR)
    print("CNN UAR:"UAR.mean())
        

if __name__ == "__main__": 

    main()