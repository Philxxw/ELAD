
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
special_symbols = {
    "[UNK]"  : 0,
    "[CLS]"  : 1,
    "[SEP]"  : 2,
    "[PAD]"  : 3,
    "[MASK]" : -float('inf'),
}
UNK_ID = special_symbols["[UNK]"]
CLS_ID = special_symbols["[CLS]"]
SEP_ID = special_symbols["[SEP]"]
MASK_ID = special_symbols["[MASK]"]

def _split_a_and_b(data, seq_len, data_dimensions,begin_idx=0,extend_target=False):
    """Split two segments from `data` starting from the index `begin_idx`."""
    '''begin_idx=i + reuse_len,
        tot_len=seq_len - reuse_len - 3
    '''
    #data_len = data.shape[0]
    cut_points = np.linspace(begin_idx+seq_len//5,begin_idx+seq_len*0.8,4)
    a_begin = begin_idx#数据a的开始位置
    end_idx = begin_idx+seq_len
    if len(cut_points) == 0 or random.random() < 0.5:
        '''
        A和B不连续，因此从找到的句子随机选一部分作为A，比
        如前两个句子，接着随机的从整个data里再寻找一部分作为B。
        '''
        # NotNext 
        label = 0
        if len(cut_points) == 0:#数据总长为0
            a_end = seq_len-1 #数据a结束位置
        else:#随机数小于0.5
            a_end = random.choice(cut_points)# 随机在每行位置选择a的结束位置
        b_len = max(1, seq_len - (a_end - a_begin))#total_len大于数据a长度的条件下，数据b长度为总长减去数据a的长度
        num = int((seq_len - b_len)//data_dimensions)
        b_begin = random.choice(np.linspace(1,num,num))*data_dimensions# 数据b开始的位置
        b_end = b_begin + b_len-1#数据b结束位置
        #找到b的起始点和终点
        
    else:
        '''
        A和B是连续的，因此从找到的句子里随机的选择前面一部分作为A，剩下的作为B。比如一共三个句子有可能前两个句子是A，后一个是B。
        '''
        # isNext
        label = 1
        a_end = random.choice(cut_points)
        b_begin = a_end
        b_end = end_idx-1
    #print('data A start from %i to %i'%(a_begin,a_end))
    #print('data B start from %i to %i'%(b_begin,b_end))
    ret = [data[int(a_begin): int(a_end)], data[int(b_begin): int(b_end)], label]
    
    if extend_target:
        '''
        if a_end >= data_len or b_end >= data_len:
            print("[_split_a_and_b] returns None: "
                          "a_end %d or b_end %d >= data_len %d",
                          a_end, b_end, data_len)
            return None'''
        a_target = data[int(a_begin)+1: int(a_end)+1]
        b_target = data[int(b_begin): int(b_end)+1]
        ret.extend([a_target,b_target])

    return ret


def _sample_mask(seg, name,len_a,data_dimensions,
                 reverse=False, goal_num_predict=None):
   
    mask = np.array([False] * seg.shape[0],dtype=np.bool)
    num_predict = 0
    if reverse:
        seg = np.flip(seg, 0)#翻转seg
    if name =='inp':
        #x = np.random.choice([0,38,76,114,152],1)
        #ngrams = np.linspace(x,x+37,38)
        ngrams = np.linspace(data_dimensions*4,data_dimensions*5-1,data_dimensions)
        cur_len = 0
        seg_len = seg.shape[0]

        #if goal_num_predict is not None and num_predict >= goal_num_predict: break
        if random.random() < 0.5:
            n = np.random.choice(ngrams,goal_num_predict,replace = False)
        else:
            n_s = np.random.choice(ngrams[0:len(ngrams)-goal_num_predict])
            n = np.linspace(n_s,n_s+goal_num_predict-1,goal_num_predict)
        for n_ in n:
            mask[int(n_)] = ~mask[int(n_)]
    elif name =='cat':
        if random.random() < 0.5:
            n1 = np.random.choice(np.linspace(len_a-data_dimensions,len_a-1,data_dimensions),goal_num_predict//2,replace = False)
            n2 = np.random.choice(np.linspace(data_dimensions*4,data_dimensions*5-2,data_dimensions-1),goal_num_predict//2,replace = False)
        else:
            n_s_1 = np.random.choice(np.linspace(len_a-data_dimensions,len_a-1-goal_num_predict//2,29),1,replace = False)
            n_s_2 = np.random.choice(np.linspace(data_dimensions*4,data_dimensions*5-1-goal_num_predict//2,29),1,replace = False)
            n1 = np.linspace(n_s_1,n_s_1+goal_num_predict//2-1,goal_num_predict//2)
            n2 = np.linspace(n_s_2,n_s_2+goal_num_predict//2-1,goal_num_predict//2)
        for n_1,n_2 in zip(n1,n2):
            mask[int(n_1)] = ~mask[int(n_1)]
            mask[int(n_2)] = ~mask[int(n_2)]
    if reverse:
        mask = np.flip(mask, 0)#翻转mask
    #print('shape of mask',np.count_nonzero(mask))
    return mask


def _create_data(data, seq_len, reuse_len,
                bi_data,data_dimensions, num_predict):
    features = []
    
    '''
    最终想要获得的数据input_data, sent_ids,
    这个过程读取每一个文件的每一行，然后使用sp切分成WordPiece，
    然后变成id(Input_data将语言文本先分词处理，然后每个词转化为数字索引，获取最终的句子)
    放到数组input_data里。另外还有一个sent_ids，用来表示句子(用整段True或整段False来标识分割不同句子)
    '''
        
    assert reuse_len < (seq_len - 1) #减两个SEP和一个CLS
    data = data.reshape(-1,data_dimensions*5)
    #data_len = data.shape[1]
    cls_array = np.array([CLS_ID])
    #标识SEP和CLS
    #print("shape of complete file:",data.shape)
    for d in range(data.shape[0]):
        inp = data[d]#data先按reuse_len长分割出固定长的数据作为cache
        tgt = data[d]#将整个data按每reuse_len长分割，相比于inp错一位
        
        #train_data
        #调用split_a_and_b
        r_n = 5
        pvals=[1/2,1/((r_n-1)*2),1/((r_n-1)*2),1/((r_n-1)*2),1/((r_n-1)*2)]
        delay = int(np.random.choice(np.linspace(1,r_n,r_n), p=pvals,replace = False))
        if d < data.shape[0]-5:
            a_b_ = data[d+delay]
        else:
            a_b_ = data[d-delay]
        
        '''
        #test_data
        if d < data.shape[0]-1:
            a_b_ = data[d+1]
        elif d == data.shape[0]-1:
            a_b_ = data[d-1]
        '''
        
        #调用split_a_and_b        
        results = _split_a_and_b(a_b_,reuse_len,data_dimensions=data_dimensions,extend_target=True)

        # unpack the results
        (a_data, b_data, label,a_target, b_target) = tuple(results)

        # sample ngram spans to predict
        reverse = bi_data#是否双向
        if num_predict is None:#num_predicts要预测的token数目，默认85
            num_predict_0 = num_predict_1 = None
        else:
            num_predict_1 = num_predict // 2
            num_predict_0 = num_predict - num_predict_1
        
        #对前面的memory先MASK
        #print('input shape',inp.shape)
        mask_0 = _sample_mask(inp,'inp',None,data_dimensions=data_dimensions,reverse=reverse,
                              goal_num_predict=num_predict_0)
        #print(mask_0.shape)
        #对后面的数据MASK
        #print('a_data shape',a_data.shape)
        #print('b_data shape',b_data.shape)
        #len_a.append(a_data.shape[0])
        mask_1 = _sample_mask(np.concatenate([a_data,b_data,cls_array]),'cat',
                              a_data.shape[0],data_dimensions=data_dimensions,
                              reverse=reverse, goal_num_predict=num_predict_1)

        # concatenate data
        '''
            cat_data
            ==================|-----| * |-----| * | +
            memorey(reuse_len)|dataA|SEP|dataB|SEP|CLS
                 mask_0       |        mask_1       
        '''
        cat_data = np.concatenate([inp, a_data, b_data, cls_array])
        
        seg_id = np.array(([0] * (reuse_len+a_data.shape[0]) +
                  [1] * b_data.shape[0] + [2]))
        
        assert cat_data.shape[0] == seq_len
        assert mask_0.shape[0] == seq_len // 2
        assert mask_1.shape[0] == seq_len // 2
        assert seg_id.shape[0] == seq_len
        # the last two CLS's are not used, just for padding purposes
        tgt = np.concatenate([tgt, a_target, b_target])

        assert tgt.shape[0] == seq_len
        is_masked = np.concatenate([mask_0, mask_1], 0)
        #print(np.sum(is_masked))
        if num_predict is not None:
            assert np.sum(is_masked) == num_predict
        feature = {
                    "input": cat_data,
                    "is_masked": is_masked,
                    "target": tgt,
                    "seg_id": seg_id,
                    "label": [label],
                    }
        features.append(feature)
    
    return features


def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
    
    # Generate permutation indices# 随机生成一个下标的排列
    index = torch.arange(seq_len, dtype=torch.int64)
    index = torch.reshape(index, [-1, perm_size]).t()
    index = index[torch.randperm(index.shape[0])]
    index = torch.reshape(index.t(), [perm_size,-1])
    
    '''上述四行代码把长度为seq_len的向量分成seq_len/perm_size段(380/190)，每段进行随机打散。'''
    '''non_func_tokens是指SEP和CLS之外的”正常”的Token，SEP，CLS位置为False'''
    #inputs: inp or (dataA|SEP|dataB|SEP|CLS)
    non_func_tokens = ~(torch.eq(inputs, SEP_ID) | torch.eq(inputs, CLS_ID))
    #print('non_func_tokens',non_func_tokens)
    '''non_mask_tokens指的是”正常”的并且没有被Mask的Token，
    即没有被Mask的”正常”的Token(non_func_tokens为True)，CLS/SEP为False'''
    #is_masked = torch.BoolTensor([False if is_masked[i]==0 else True for i in range(len(is_masked))])
    non_mask_tokens = (~is_masked) & non_func_tokens#mask的位置置为False
    #print('non_mask_tokens',non_mask_tokens)
    '''masked_or_func_tokens和non_mask_tokens相反，包括Masked的Token和SEP与CLS'''
    masked_or_func_tokens = ~non_mask_tokens 

    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss 
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = masked_or_func_tokens & non_func_tokens
    target_mask = target_tokens.type(torch.float32)

    # new target: [next token] for LM and [curr token] (self) for PLM
    '''对于常规的语言模型来说，我们是预测下一个词，而XLNet是根据之前的状态和当前的位置预测被MASK的当前词。
    所以真正的new_targets要前移一个。'''
    new_targets = torch.cat([inputs[0: 1], targets[: -1]], dim=0)

    # construct inputs_k
    inputs_k = inputs
    
    # construct inputs_q
    inputs_q = target_mask

    return  new_targets, target_mask, inputs_k, inputs_q


def make_permute(feature,reuse_len, seq_len, perm_size, num_predict):

    inputs = torch.LongTensor(feature.pop("input"))
    target = torch.LongTensor(feature.pop("target"))
    is_masked = torch.BoolTensor(feature.pop("is_masked"))
    
    non_reuse_len = seq_len - reuse_len
    assert perm_size <= reuse_len and perm_size <= non_reuse_len

    target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
        inputs[:reuse_len], # inp
        target[:reuse_len],
        is_masked[:reuse_len],
        perm_size,
        reuse_len)

    target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
        inputs[reuse_len:], # (senA, senBm cls)
        target[reuse_len:],
        is_masked[reuse_len:],
        perm_size,
        non_reuse_len)
    
    #perm_mask_0 = torch.cat([perm_mask_0, torch.ones([reuse_len, non_reuse_len,190])],
    #                        dim=1)
    #perm_mask_1 = torch.cat([torch.zeros([non_reuse_len, reuse_len,190]), perm_mask_1],
    #                        dim=1)
    #perm_mask = torch.cat([perm_mask_0, perm_mask_1], dim=0)

    target = torch.cat([target_0, target_1], dim=0)
    target_mask = torch.cat([target_mask_0, target_mask_1], dim=0)
    input_k = torch.cat([input_k_0, input_k_1], dim=0)
    input_q = torch.cat([input_q_0, input_q_1], dim=0)
    '''
    print('perm_mask.shape',perm_mask.shape)
    print('target.shape',target.shape)
    print('target_mask.shape',target_mask.shape)
    print('input_k.shape',input_k.shape)
    print('input_q.shape',input_q.shape)
    '''
    if num_predict is not None:
        indices = torch.arange(seq_len, dtype=torch.int64)
        bool_target_mask = target_mask.byte()
        indices = indices[bool_target_mask]
        ##### extra padding due to CLS/SEP introduced after prepro
        ''' 
        # 因为随机抽样的MASK可能是CLS/SEP，这些是不会被作为预测值的，因此
        # 我们之前生成的数据有num_predict个需要预测的，但实际需要预测的只有actual_num_predict
        # 所以还需要padding=num_predict - actual_num_predict个
        '''
        actual_num_predict = indices.shape[0]
        pad_len = num_predict - actual_num_predict
        assert seq_len >= actual_num_predict

        ##### target_mapping
        
        target_mapping = torch.eye(seq_len, dtype=torch.float32)[indices]#被mask的位置用one-hot标识
        paddings = torch.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
        target_mapping = torch.cat([target_mapping, paddings], dim=0)
        feature["target_mapping"] = torch.reshape(target_mapping,
                                                [num_predict, seq_len])
        
        ##### target
        target = target[bool_target_mask]
        paddings = torch.zeros([pad_len], dtype=target.dtype)
        target = torch.cat([target, paddings], dim=0)
        feature["target"] = torch.reshape(target, [num_predict])

        ##### target mask
        target_mask = torch.cat(
            [torch.ones([actual_num_predict], dtype=torch.float32),
             torch.zeros([pad_len], dtype=torch.float32)],
            dim=0)
        feature["target_mask"] = torch.reshape(target_mask, [num_predict])
    else:
        feature["target"] = torch.reshape(target, [seq_len])
        feature["target_mask"] = torch.reshape(target_mask, [seq_len])

    # reshape back to fixed shape
    # 1. perm_mask，64x64，表示经过重新排列后第i个token能否attend to 第j个token，1表示不能attend
    # 2. target，64，表示真实的目标值，之前生成的target是预测下一个词，但是XLNet是预测当前词
    # 3. target_mask，64，哪些地方是Mask的(需要预测的)
    # 4. input_k, 64，content stream的初始值
    # 5. input_q, 64, 哪些位置是需要计算loss的，如果不计算loss，也就不计算Query Stream。
    feature["seg_id"] = torch.IntTensor(feature["seg_id"])
    #feature["perm_mask"] = torch.reshape(perm_mask, [seq_len, seq_len])
    feature["input_k"] = torch.reshape(input_k, [seq_len])
    feature["input_q"] = torch.reshape(input_q, [seq_len])
    
    return feature

if __name__ == "__main__":
    files_data =['MSL']
    #SMAP:
    #['SMAP'] 
    #SMD:
    #    ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
    #     'machine-2-1','machine-2-2','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7','machine-2-8',
    #     'machine-2-9','machine-3-1','machine-3-2','machine-3-3','machine-3-4','machine-3-5','machine-3-6','machine-3-7',
    #     'machine-3-8','machine-3-9','machine-3-10','machine-3-11']

    input_index_path='prepare_input_MSL/'
    #'prepare_input_SMAP/'#'prepare_input_SMD/'
    num_predict=45 #MSL 45,SMAP 25, SMD38
    data_dimensions = 55#MSL 55, SMAP 25, SMD38
    
    for f in files_data:
        permutations = []
        #SMD #input_index_path+‘test_tocken/’+f+'_train_tocken_input.csv'
        input_data = np.loadtxt(input_index_path+f+'_test_tocken_input.csv',delimiter=',')
        
        data = np.array([input_data], dtype=np.int64)
        ##SMD #input_index_path+‘train_tocken/’+f+'_train_tocken_input.csv'
        train_data = np.loadtxt(input_index_path+f+'_train_tocken_input.csv',delimiter=',')
        data = data[:,0:train_data.shape[0]]
        features = _create_data(data = data,
                            seq_len=data_dimensions*10,
                            reuse_len=data_dimensions*5,
                            data_dimensions=data_dimensions,
                            bi_data=False,
                            num_predict=num_predict)
        
        c = 0
        for feature in features:
            '''
            c += 1
            if c ==1:
                print('input',feature["input"])
                print('is_masked',feature["is_masked"])
                print('target',feature["target"])
                print('seg_id',feature["seg_id"])'''
            permutation = make_permute(feature,
                               reuse_len=data_dimensions*5,
                               seq_len=data_dimensions*10,
                               perm_size=data_dimensions*5,
                               num_predict=num_predict)
            permutations.append(permutation)
            '''
            if c ==1:
                print('target_permutation',permutation['target'])
                print('target_mask_permutation',permutation['target_mask'])
                print('input_k_permutation',permutation['input_k'])
                print('input_q_permutation',permutation['input_q'])'''
        print(len(permutations))
        print(f)
        np.save(input_index_path+"test_data/%s.npy"%(f),permutations)

