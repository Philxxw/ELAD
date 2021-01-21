
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import data_utils
import argparse
import os
import model
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
#from pytorch_pretrained_bert import BertTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
''''''
class permutations_data(Dataset):
    def __init__(self,permutation_data):
        self.permutation_data =permutation_data
    def __getitem__(self, item):
        inp_k =   self.permutation_data[item]['input_k'].float()
        seg_id =  self.permutation_data[item]['seg_id'].float()
        target =  self.permutation_data[item]['target'].float()
        target_mapping =  self.permutation_data[item]['target_mapping'].float()
        inp_q =  self.permutation_data[item]['input_q'].float()
        tgt_mask =  self.permutation_data[item]['target_mask'].float()

        inp_k = torch.Tensor(inp_k)
        seg_id = torch.Tensor(seg_id)
        target = torch.Tensor(target)
        target_mapping = torch.Tensor(target_mapping)
        inp_q = torch.Tensor(inp_q)
        tgt_mask = torch.Tensor(tgt_mask)
        
        return inp_k,seg_id,target,target_mapping,inp_q,tgt_mask

    def __len__(self):
        return len(self.permutation_data)
    
if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description='PyTorch Model')
    parser.add_argument('--data', type=str, default='prepare_input_SMD/')
    parser.add_argument('--seq_len', type=int, default=38*10, help="Sequence length.")
    parser.add_argument('--reuse_len', type=int, default=38*5,
                        help="Number of token that can be reused as memory. "
                             "Could be half of `seq_len`.")
    parser.add_argument('--perm_size', type=int,
                        default=38*5,
                        help="the length of longest permutation. Could be set to be reuse_len.")
    parser.add_argument('--bi_data', type=bool, default=False,
                        help="whether to create bidirectional data")
    parser.add_argument('--num_predict', type=int,
                        default=32, help="Num of tokens to predict.")
    parser.add_argument('--mem_len', type=int,
                        default=38*5, help="Number of steps to cache")
    parser.add_argument('--num_epoch', type=int,
                        default=15, help="Number of epochs")

    args = parser.parse_args()
    
    bsz =16
    input_index_path=args.data+'train_data/'
    #files_data = ['SMAP']
    #files_data = ['MSL']
    files_data = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7',
                  'machine-1-8','machine-2-1','machine-2-2','machine-2-3','machine-2-4','machine-2-7','machine-2-8',
                  'machine-2-9','machine-3-1','machine-3-2','machine-3-3','machine-3-4','machine-3-5','machine-3-6',
                  'machine-3-7','machine-3-8','machine-3-9','machine-3-10','machine-3-11']

    for f in files_data:  
        data = np.load(input_index_path+f+'.npy',allow_pickle=True)
        print(f.split('.',1)[0])
        #SMAP#MSL
        #eb_dicts = torch.Tensor(np.loadtxt(args.data+f+'_train_embedding_dict.csv', delimiter =','))
        #SMD
        eb_dicts = torch.Tensor(np.loadtxt(args.data+'train_eb_dict/'+f+'_train_embedding_dict.csv', delimiter =','))
        print(eb_dicts.shape[0])
        #print(EB_len.shape[0])
        model = model.Model(n_layer=6, n_head=4, d_head=8,
                        d_inner=32, d_model=32,
                        dropout=0.1, dropatt=0.1,
                        attn_type="bi", bi_data=args.bi_data,
                        clamp_len=-1, same_length=False,
                        reuse_len=args.reuse_len, mem_len=args.mem_len,
                        embedding_dict = eb_dicts).cuda()
        model.train()
        criterion = nn.MSELoss().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.01)
        
        tr_dataset = permutations_data(data)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=bsz, shuffle=False,num_workers=0)
        for num_epoch in range(args.num_epoch):
            num_step = 0
            mems = None
            t1 = time.time()
            for tr_data in tr_dataloader:
                inp_k,seg_id,target,target_mapping,inp_q,tgt_mask = tr_data[0],tr_data[1],tr_data[2],tr_data[3],tr_data[4],tr_data[5]
                '''
                inp_k,# [seq_len, 1(=bsz)]
                seg_id,# [seq_len, 1(=bsz)]
                target,#[num_predict, 1(=bsz)] 
                target_mapping,#[num_predict, seq_len, 1(=bsz)]
                inp_q,#[seq_len, 1(=bsz)]
                tgt_mask #tgt_mask [num_predict, 1(=bsz)]
                '''
                perm_mask =None

                logits, new_mems,sequence_output = model(inp_k=torch.reshape(inp_k,[args.seq_len,-1]).cuda(), 
                                         seg_id=torch.reshape(seg_id,[args.seq_len,-1]).cuda(),
                                         input_mask=None,mems=mems, perm_mask=perm_mask,
                                         target_mapping=torch.reshape(target_mapping,[args.num_predict,args.seq_len,-1]).cuda(),
                                         inp_q=torch.reshape(inp_q,[args.seq_len,-1]).cuda())
                
                input_ = torch.reshape(eb_dicts[inp_k.long()],[args.seq_len,-1,32])
                target = torch.reshape(eb_dicts[target.long()],[args.num_predict,-1,32])
                
                lm_loss_input = criterion(sequence_output[38*4:-1].cuda(), input_[38*4:-1].cuda()).type(torch.float32).cuda()
                #lm_loss_target = criterion(logits.cuda(), target.cuda()).type(torch.float32).cuda()
                lm_loss = lm_loss_input#+lm_loss_target
                
                optimizer.zero_grad()
                num_step += 1

                lm_loss.backward()
                optimizer.step()

                t2 = time.time()
                print('Number of Epoch: %04d in %04d Step' % ((num_epoch + 1), (num_step + 1)),
                      'cost =', '{:.6f}'.format(lm_loss),
                      'time cost =', '{:.6f}'.format(t2-t1))
                
                mems = new_mems
        torch.save(model.state_dict(), args.data+'model_par/'+f+'.pkl')
