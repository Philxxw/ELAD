from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
        Defines a Transformer-XL computation graph.

        Args:

        inp_k: int32 Tensor in shape [len, bsz], the input token IDs.
        seg_id: int32 Tensor in shape [len, bsz], the input segment IDs.
        input_mask: float32 Tensor in shape [len, bsz], the input mask.
          0 for real tokens and 1 for padding.
        mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
          from previous batches. The length of the list equals n_layer.
          If None, no memory is used.
        perm_mask: float32 Tensor in shape [len, len, bsz].
          If perm_mask[i, j, k] = 0, i attend to j in batch k;
          if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
          If None, each position attends to all the others.
        target_mapping: float32 Tensor in shape [num_predict, len, bsz].
          If target_mapping[i, j, k] = 1, the i-th predict in batch k is
          on the j-th token.
          Only used during pretraining for partial prediction.
          Set to None during finetuning.
        inp_q: float32 Tensor in shape [len, bsz].
          1 for tokens with losses and 0 for tokens without losses.
          Only used during pretraining for two-stream attention.
          Set to None during finetuning.

        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        ff_activation: str, "relu" or "gelu".


        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.

        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
          and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
          Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
          -1 means no clamping.

      """
    def __init__(self, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, reuse_len, mem_len,embedding_dict):
        super(Model, self).__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.d_model = d_model
        self.dropout = dropout
        self.dropatt = dropatt
        self.attn_type = attn_type
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.reuse_len = reuse_len
        self.mem_len = mem_len
        
        self.embedding = embedding_dict.cuda()
        self.Dropout = nn.Dropout(p=dropout)
        self.DropAttn = nn.Dropout(p=dropatt)

        self.r_w_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head,self.d_head)).cuda()#u(相对位置编码)
        self.r_r_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head, self.d_head)).cuda()#v(相对位置编码)

        ##### Segment embedding
        self.r_s_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head,self.d_head)).cuda()

        self.seg_embed = nn.Parameter(torch.randn(self.n_layer, 2,
                                                   self.n_head, self.d_head)).cuda()

        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_model)).cuda()

        # post-attention projection (back to `d_model`)
        self.proj_o = nn.Parameter(torch.randn(self.d_model,
                                                self.n_head, self.d_head)).cuda()

        #### Project hidden states to a specific head with a 4D-shape.
        self.q_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head)).cuda()
        self.k_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head)).cuda()
        self.v_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head)).cuda()
        self.r_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head)).cuda()

        self.layer_norm = nn.LayerNorm(d_model)

        self.conv1 = nn.Linear(d_model, d_inner)
        self.conv2 = nn.Linear(d_inner, d_model)
        self.relu = nn.ReLU(inplace=True)

        self.softmax_b = nn.Parameter(torch.zeros(embedding_dict.shape[0]))


    def gelu(self, x):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.

        Returns:
          `x` with the GELU activation applied.
        """
        cdf =( 0.5 * (1.0 + torch.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))).cuda()
        return x.cuda() * cdf

    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        '''x=bd，klen = ac.shape[1]
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)
        '''
        x_size = x.shape
        x = torch.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])

        x = x[1:, 0:, 0:, 0:] # tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        # reshape
        x = torch.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
        # slice
        x = x[0:, 0:klen, 0:, 0:].cuda() # tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

        return x

    def positionwise_ffn(self, inp, activation_type='relu'):

        """Position-wise Feed-forward Network"""
        output = self.conv1(inp)
        output = self.Dropout(output)
        if activation_type == 'relu':
            output = self.relu(output)
        elif activation_type == 'gelu':
            output = self.gelu(output)
        else:
            raise ValueError('Unsupported activation type {}'.format(activation_type))

        output = self.layer_norm(output + inp)
        return output.cuda()

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing.
        """

        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.proj_o)

        attn_out = self.Dropout(attn_out)
        if residual:
            output = self.layer_norm(attn_out + h)
        else:
            output = self.layer_norm(attn_out)

        return output.cuda()

    def head_projection(self, h, name):
        """Project hidden states to a specific head with a 4D-shape."""
        proj_weight = None#d_model,n_head,d_head
        
        if name == 'q':
            proj_weight = self.q_proj_weight
        elif name == 'k':
            proj_weight = self.k_proj_weight
        elif name =='v':
            proj_weight = self.v_proj_weight
        elif name == 'r':
            proj_weight = self.r_proj_weight
        else:
            raise ValueError('Unknown `name` {}.'.format(name))
        
        head = torch.einsum('ibh,hnd->ibnd', h.cuda(), proj_weight.cuda())

        return head.cuda()

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                      r_w_bias, r_r_bias, r_s_bias, attn_mask, scale):

        """Core relative positional attention operations.
        """

        # content based attention score
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h).cuda()

        #print('q_head.shape:',q_head.shape)
        #print('r_r_bias.shape:',r_r_bias.shape)
        #print('k_head_r:',k_head_r.shape)
        # position based attention score
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r).cuda()
        bd = self.rel_shift(bd, klen=ac.shape[1]).cuda()

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
#            print('q_head+r_s_bias',(q_head+r_s_bias).shape)
#            print('seg_embed',seg_embed.shape)
            ef = torch.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed).cuda()
            #print('seg_mat.shape',seg_mat.shape)
            #print('ef.shape',ef.shape)
            ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef).cuda()

        # merge attention scores and perform masking
        #print('ac',ac.shape)
        #print('bd',bd.shape)
        #print('ef',ef.shape)
        attn_score = (ac + bd + ef) * scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e30 * attn_mask
        # attention probability

        attn_prob = F.softmax(attn_score.cuda(), dim=1)
        attn_prob = self.DropAttn(attn_prob.cuda())

        # attention output

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h).cuda()
        #
        return attn_vec.cuda()

    def rel_multihead_attn(self, h, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                           attn_mask, mems, d_model, n_head, d_head, dropout, dropatt):
        """Multi-head attention with relative positional encoding."""

        scale = 1 / (d_head ** 0.5)
        if mems is not None and len(mems.size()) > 1:
            if mems.shape[1] != h.shape[1]: 
                mems= mems[:,mems.shape[1]-h.shape[1]-1:-1,:]
            cat = torch.cat([mems, h], dim=0).cuda()
        else:
            cat = h.cuda()

        # content heads
        q_head_h = self.head_projection(h.cuda(), 'q')
        k_head_h = self.head_projection(cat.cuda(), 'k')
        v_head_h = self.head_projection(cat.cuda(), 'v')

        # positional heads
        k_head_r = self.head_projection(r.cuda(), 'r')

        # core attention ops
        attn_vec = self.rel_attn_core(
            q_head_h.cuda(), k_head_h.cuda(), v_head_h.cuda(), k_head_r.cuda(), seg_embed.cuda(), seg_mat.cuda(), r_w_bias.cuda(),
            r_r_bias.cuda(), r_s_bias.cuda(), attn_mask.cuda(), scale)

        # post processing
        output = self.post_attention(h.cuda(), attn_vec.cuda())

        return output.cuda()

    def two_stream_rel_attn(self, h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias,
                            seg_embed, attn_mask_h, attn_mask_g, target_mapping):
        
        """Two-stream attention"""
        scale = 1 / (self.d_head ** 0.5)

        # content based attention score
        if mems is not None and len(mems.size()) > 1:
            if mems.shape[1] != h.shape[1]: 
                mems= mems[:,mems.shape[1]-h.shape[1]-1:-1,:]
            cat = torch.cat([mems, h], dim=0).cuda()
        else:
            cat = h.cuda()

        # content-based key head
        k_head_h = self.head_projection(cat.cuda(), 'k')

        # content-based value head
        v_head_h = self.head_projection(cat.cuda(), 'v')

        # position-based key head
        k_head_r = self.head_projection(r.cuda(), 'r')

        ##### h-stream
        # content-stream query head
        q_head_h = self.head_projection(h.cuda(), 'q')

        # core attention ops
        # hˆ(m)_zt = LayerNorm(h^(m-1)_zt + RelAttn(h^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        # 输出是attention score
        attn_vec_h = self.rel_attn_core(
            q_head_h.cuda(), k_head_h.cuda(), v_head_h.cuda(), k_head_r.cuda(), seg_embed.cuda(), seg_mat.cuda(), r_w_bias.cuda(),
            r_r_bias.cuda(), r_s_bias.cuda(), attn_mask_h, scale)

        # post processing
        output_h = self.post_attention(h.cuda(), attn_vec_h.cuda())

        ##### g-stream
        # query-stream query head
        q_head_g = self.head_projection(g.cuda(), 'q')

        # core attention ops
        # gˆ(m)_zt = LayerNorm(g^(m-1)_zt + RelAttn(g^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        if target_mapping is not None:
            #print('q_head_g',q_head_g.shape)
            #print('target_mapping',target_mapping.shape)
            q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g.cuda(), target_mapping.cuda()).cuda()
            # Attention
            attn_vec_g = self.rel_attn_core(
                q_head_g.cuda(), k_head_h.cuda(), v_head_h.cuda(), k_head_r.cuda(), seg_embed.cuda(), seg_mat.cuda(), r_w_bias.cuda(),
                r_r_bias.cuda(), r_s_bias.cuda(), attn_mask_g, scale)

            attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g.cuda(), target_mapping.cuda())
        else:
            attn_vec_g = self.rel_attn_core(
                q_head_g.cuda(), k_head_h.cuda(), v_head_h.cuda(), k_head_r.cuda(), seg_embed.cuda(), seg_mat.cuda(), r_w_bias.cuda(),
                r_r_bias.cuda(), r_s_bias.cuda(), attn_mask_g.cuda(), scale)

        # post processing

        output_g = self.post_attention(g.cuda(), attn_vec_g.cuda())
        return output_h.cuda(), output_g.cuda()


    def _create_mask(self, qlen, mlen, dtype, same_length=False):
        """create causal attention mask."""
        # [[0,1,1],
        #  [0,0,1],
        #  [0,0,0]]
        attn_mask = torch.ones([qlen, qlen], dtype=dtype).cuda()
        mask_u = torch.triu(attn_mask).cuda() # Upper triangular part.
        mask_dia = torch.tril(attn_mask).cuda() & torch.triu(attn_mask).cuda() # Diagonal. Figure 2(c)
        attn_mask_pad = torch.zeros([qlen, mlen], dtype=dtype).cuda()
        ret = torch.cat([attn_mask_pad, mask_u - mask_dia], dim=1).cuda() # [qlen, mlen]
        if same_length:
            # [[0,1,1],
            #  [1,0,1],
            #  [1,1,0]]
            mask_l = torch.tril(attn_mask).cuda() # Lower triangular part.
            ret = torch.cat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], dim=1).cuda()

        return ret.type(dtype=torch.float32) # [qlen, qlen]

    def positional_embedding(self, pos_seq, inv_freq,bsz = None):

        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq).cuda()
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1).cuda()
        pos_emb = pos_emb[:, None, :].cuda()
        if bsz is not None:

            #print('pos_emb.shape',pos_emb.repeat(1,bsz,1).shape)
            return pos_emb.repeat(1, bsz, 1).cuda()
            
        return pos_emb.cuda()

    def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None):
        """cache hidden states into memory."""
        """把隐状态chache进memory"""
        with torch.no_grad():          

            if mem_len is None or mem_len == 0:
                return None
            else:
                if reuse_len is not None and reuse_len > 0:
                    curr_out = curr_out[:reuse_len].cuda()

                if prev_mem is None:
                    
                    new_mem = curr_out[-mem_len:].cuda()
                else:
                    #print(prev_mem[:,prev_mem.shape[1]-curr_out.shape[1]-1:-1,:].shape)
                    if curr_out.shape != prev_mem.shape:
                        #print('curr_out.shape',curr_out.shape)
                        #print('prev_mem.shape',prev_mem.shape)
                        prev_mem= prev_mem[:,prev_mem.shape[1]-curr_out.shape[1]-1:-1,:]
                        #print('prev_mem.shape',prev_mem.shape)
  
                    new_mem = torch.cat([prev_mem, curr_out], dim=0)[-mem_len:].cuda()
                    
            return new_mem.cuda()


    def relative_positional_encoding(self, qlen, klen, d_model, clamp_len, attn_type,
                                     bi_data, bsz=None, dtype=None):
        """create relative positional encoding."""
        # 长度为d_model/2
        freq_seq = torch.arange(0, d_model, 2.0)
        if dtype is not None and dtype != torch.float32:
            freq_seq = freq_seq.type(dtype)
        inv_freq = 1 / (10000 ** (freq_seq / d_model))

        if attn_type == 'bi':
            # beg, end = klen - 1, -qlen

            beg, end = klen, -qlen
        elif attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

        if bi_data and bsz%2 == 0:
            fwd_pos_seq = torch.arange(beg, end, -1.0).cuda()
            bwd_pos_seq = torch.arange(-beg, -end, 1.0).cuda()

            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
                bwd_pos_seq = bwd_pos_seq.type(dtype=dtype)

            if clamp_len > 0:

                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len).cuda()
                bwd_pos_seq = torch.clamp(bwd_pos_seq, -clamp_len, clamp_len).cuda()

            fwd_pos_emb = self.positional_embedding(fwd_pos_seq.cuda(), inv_freq.cuda(), bsz//2)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq.cuda(), inv_freq.cuda(), bsz//2)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1).cuda()
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0).cuda()
            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
            if clamp_len > 0:
                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len).cuda()
            if bsz is not None:
                pos_emb = self.positional_embedding(fwd_pos_seq.cuda(), inv_freq.cuda(), bsz)
            else:
                pos_emb = self.positional_embedding(fwd_pos_seq.cuda(), inv_freq.cuda())

        return pos_emb

    def forward(self, inp_k, seg_id, input_mask, mems, perm_mask, target_mapping, inp_q):
        new_mems = []

        bsz = inp_k.shape[1]
        qlen = inp_k.shape[0]
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self._create_mask(qlen, mlen, torch.int64, self.same_length).cuda()
            attn_mask = attn_mask[:, :, None, None].cuda()
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None].cuda()
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz],
                                 dtype=torch.float32).cuda()
            data_mask = torch.cat([mems_mask, data_mask], dim=1).cuda()
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None].cuda()
            else:
                attn_mask += data_mask[:, :, :, None].cuda()

        if attn_mask is not None:
            attn_mask = attn_mask.gt(0).type(torch.float32).cuda()

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen, dtype=torch.float32) # [qlen, qlen]
            non_tgt_mask = torch.cat([torch.zeros([qlen, mlen], dtype=torch.float32), # [qlen, klen]
                                        non_tgt_mask],
                                        dim=-1).cuda()
            non_tgt_mask = (attn_mask.cuda() +
                            non_tgt_mask[:, :, None, None].cuda()).gt(0).type(dtype=torch.float32)
        else:
            non_tgt_mask = None

        ##### Word embedding
        lookup_table = self.embedding
        word_emb_k = lookup_table[inp_k.type(torch.LongTensor)].cuda()

        if inp_q is not None:#
            if target_mapping is not None:
                word_emb_q = self.mask_emb.repeat(target_mapping.shape[0], bsz, 1).cuda()
            else:

                inp_q_ext = inp_q[:, :, None].cuda()
                word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k     
        #### Figure 2(a), Content Stream(Original Attention), h^(0)_t = e(x_i) = e(inp_k)
        output_h = self.Dropout(word_emb_k)
        if inp_q is not None:##
            #### Query Stream, g^(0)_t = w
            #### the first layer query stream is initialized with a trainable vector
            output_g = self.Dropout(word_emb_q)

        ##### Segment embedding
        # paper
        # Given a pair of positions i and j in the sequence, if
        # i and j are from the same segment
        if seg_id is not None:
            # Convert `seg_id` to one-hot `seg_mat`
            #print(mlen,bsz)
            mem_pad = torch.zeros([mlen, bsz], dtype=torch.int32)
            seg_id = torch.reshape(seg_id.type(torch.IntTensor),[-1,bsz])
            cat_ids = torch.cat([mem_pad, seg_id], dim=0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (~torch.eq(seg_id[:, None], cat_ids[None, :])).type(torch.long)
            seg_mat = torch.eye(2, dtype=torch.float32)[seg_mat]
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(
            qlen, klen, self.d_model, self.clamp_len, self.attn_type, self.bi_data,
            bsz=bsz, dtype=torch.float32).cuda()
        pos_emb = self.Dropout(pos_emb)

        ##### Attention layers
        if mems is None:
            mems = [None] * self.n_layer

        for i in range(self.n_layer):
            # cache new mems
            new_mems.append(self._cache_mem(output_h, mems[i], self.mem_len, self.reuse_len).cuda())

            # segment bias
            if seg_id is None:
                r_s_bias_i = None
                seg_embed_i = None
            else:
                r_s_bias_i = self.r_s_bias[i].cuda()
                seg_embed_i = self.seg_embed[i].cuda()

            if inp_q is not None:###
                output_h, output_g = self.two_stream_rel_attn(
                    h=output_h.cuda(),
                    g=output_g.cuda(),
                    r=pos_emb.cuda(),
                    r_w_bias= self.r_w_bias[i].cuda(),
                    r_r_bias= self.r_r_bias[i].cuda(),
                    seg_mat=seg_mat.cuda(),
                    r_s_bias=r_s_bias_i.cuda(),
                    seg_embed=seg_embed_i.cuda(),
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    mems=mems[i],
                    target_mapping=target_mapping.cuda())
            else:
                output_h = self.rel_multihead_attn(
                    h=output_h.cuda(),
                    r=pos_emb.cuda(),
                    r_w_bias=self.r_w_bias[i].cuda(),
                    r_r_bias=self.r_r_bias[i].cuda(),
                    seg_mat=seg_mat.cuda(),
                    r_s_bias=r_s_bias_i.cuda(),
                    seg_embed=seg_embed_i.cuda(),
                    attn_mask=non_tgt_mask,
                    mems=mems[i])
                

            if inp_q is not None:
                output_g = self.positionwise_ffn(inp=output_g.cuda()).cuda()
            output_h = self.positionwise_ffn(inp=output_h.cuda()).cuda()
        if inp_q is not None:
            output = self.Dropout(output_g)
        else:
            output = self.Dropout(output_h)

        #print('output',output)
        #print('output_h',output_h.shape)
        #logits = torch.einsum('ibd,nd->ibn', output.cuda(), lookup_table.cuda()) + self.softmax_b.cuda()
        
        return output, new_mems,output_h


