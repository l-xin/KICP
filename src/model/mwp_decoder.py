# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

# GTS decoder for math word problems
# From "A Goal-Driven Tree-Structured Neural Model for Math Word Problems" (IJCAI 2019)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        return

    def forward(self, query, context, mask=None):
        # query/context: batch_size * seq_len * hidden_dim
        # mask: batch_size * seq_len
        batch_size, query_size, hidden_dim = query.size()
        context_size = context.size(1)
        # batch_size * query_size * context_size * hidden_dim
        qc_query = query.unsqueeze(2).expand(-1, -1, context_size, -1)
        qc_context = context.unsqueeze(1).expand(-1, query_size, -1, -1)
        score_hidden = torch.cat((qc_query, qc_context), dim=-1)
        score_hidden = F.leaky_relu(self.w(score_hidden))
        score = self.score(score_hidden).view(batch_size, query_size, context_size)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, query_size, -1)
            score.masked_fill_(mask==0, -float('inf'))
        attn = F.softmax(score, dim=-1)
        # (b, q, c) * (b, c, d) -> (b, q, d)
        attn_output = torch.bmm(attn, context)
        # attn_output: (b, q, d)
        # attn  : (b, q, c)
        return attn_output, attn

class GateNN(nn.Module):
    def __init__(self, hidden_dim, input1_dim, input2_dim=0, dropout=0.4, one_layer=False):
        super(GateNN, self).__init__()
        self.one_layer = one_layer
        self.hidden_l1 = nn.Linear(input1_dim+hidden_dim, hidden_dim)
        self.gate_l1 = nn.Linear(input1_dim+hidden_dim, hidden_dim)
        if not self.one_layer:
            self.dropout = nn.Dropout(p=dropout)
            self.hidden_l2 = nn.Linear(input2_dim+hidden_dim, hidden_dim)
            self.gate_l2 = nn.Linear(input2_dim+hidden_dim, hidden_dim)
        return
    
    def forward(self, hidden, input1, input2=None):
        input1 = torch.cat((hidden, input1), dim=-1)
        h = torch.tanh(self.hidden_l1(input1))
        g = torch.sigmoid(self.gate_l1(input1))
        h = h * g
        if not self.one_layer:
            h1 = self.dropout(h)
            if input2 is not None:
                input2 = torch.cat((h1, input2), dim=-1)
            else:
                input2 = h1
            h = torch.tanh(self.hidden_l2(input2))
            g = torch.sigmoid(self.gate_l2(input2))
            h = h * g
        return h

class ScoreModel(nn.Module):
    def __init__(self, hidden_dim):
        super(ScoreModel, self).__init__()
        self.w = nn.Linear(hidden_dim * 3, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        return
    
    def forward(self, hidden, context, token_embeddings):
        # hidden/context: batch_size * hidden_dim
        # token_embeddings: batch_size * class_size * hidden_dim
        batch_size, class_size, _ = token_embeddings.size()
        hc = torch.cat((hidden, context), dim=-1)
        # (b, c, h)
        hc = hc.unsqueeze(1).expand(-1, class_size, -1)
        hidden = torch.cat((hc, token_embeddings), dim=-1)
        hidden = F.leaky_relu(self.w(hidden))
        score = self.score(hidden).view(batch_size, class_size)
        return score

class PredictModel(nn.Module):
    def __init__(self, hidden_dim, class_size, op_size, dropout=0.4):
        super(PredictModel, self).__init__()
        self.class_size = class_size

        self.attn = Attention(hidden_dim)
        self.score_num = ScoreModel(hidden_dim)
        self.score_op = nn.Linear(hidden_dim * 2, op_size)
        self.dropout = nn.Dropout(p=dropout)
        return
    
    def score(self, hidden, context, class_embedding_masks, finished_mask):
        # embedding: batch_size * num_size * hidden_dim
        # mask: batch_size * num_size
        # batch_size * symbol_size
        # const + num
        num_embedding, num_mask, _ = class_embedding_masks
        hidden = self.dropout(hidden)
        context = self.dropout(context)
        num_embedding = self.dropout(num_embedding)
        num_score = self.score_num(hidden, context, num_embedding)
        num_score.masked_fill_(~num_mask, -float('inf'))
        # op
        op_score = self.score_op(torch.cat((hidden, context), dim=-1))
        # op + const + num
        score = torch.cat((op_score, num_score), dim=-1)
        score = F.log_softmax(score, dim=-1)

        # pad
        # finished_mask = 0, => unfinished => pad_score = -inf, others > -inf
        # finished_mask = 1 => finished => pad_score = 0, others = -inf
        finished_mask = finished_mask.unsqueeze(-1)
        # 0 => -inf, 1 => 0
        pad_score = (1 - finished_mask) * (-float('inf'))
        finished_mask = finished_mask == 1
        score = score.masked_fill(finished_mask, -float('inf'))
        # pad + op + const + num
        score = torch.cat((pad_score, score), dim=-1)
        return score

    def forward(self, node_hidden, encoder_outputs, encoder_masks, embedding_masks, finished_mask):
        dp_hidden = self.dropout(node_hidden)
        output_attn, _ = self.attn(dp_hidden.unsqueeze(1), encoder_outputs, mask=encoder_masks)
        context = output_attn.squeeze(1)

        # log_softmax
        score = self.score(node_hidden, context, embedding_masks, finished_mask)
        # batch_size * class_size
        # pad + op + const + num + empty_num
        pad_size = self.class_size - score.size(-1)
        pad_empty_num = torch.ones(score.size(0), pad_size, device=score.device) * (-float('inf'))
        score = torch.cat((score, pad_empty_num), dim=-1)
        return score, context

class TreeEmbeddingNode:
    def __init__(self, embedding, terminal):
        self.embedding = embedding
        self.terminal = terminal
        return

class TreeEmbeddingModel(nn.Module):
    def __init__(self, hidden_dim, op_set, dropout=0.4):
        super(TreeEmbeddingModel, self).__init__()
        self.op_set = op_set
        self.dropout = nn.Dropout(p=dropout)
        self.combine = GateNN(hidden_dim, hidden_dim * 2, dropout=dropout, one_layer=True)
        return
    
    def merge(self, op_embedding, left_embedding, right_embedding):
        te_input = torch.cat((left_embedding, right_embedding), dim=-1)
        te_input = self.dropout(te_input)
        op_embedding = self.dropout(op_embedding)
        tree_embed = self.combine(op_embedding, te_input)
        return tree_embed
    
    def serial_forward(self, class_embedding, tree_stacks, embed_node_index):
    # def forward(self, class_embedding, tree_stacks, embed_node_index):
        # embed_node_index: batch_size
        batch_index = torch.arange(embed_node_index.size(0), device=embed_node_index.device)
        labels_embedding = class_embedding[batch_index, embed_node_index]
        for node_label, tree_stack, label_embedding in zip(embed_node_index.cpu().tolist(), tree_stacks, labels_embedding):
            # operations
            if node_label in self.op_set:
                tree_node = TreeEmbeddingNode(label_embedding, terminal=False)
            # numbers
            else:
                right_embedding = label_embedding
                # on right tree => merge
                while len(tree_stack) >= 2 and tree_stack[-1].terminal and (not tree_stack[-2].terminal):
                    left_embedding = tree_stack.pop().embedding
                    op_embedding = tree_stack.pop().embedding
                    right_embedding = self.merge(op_embedding, left_embedding, right_embedding)
                tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
            tree_stack.append(tree_node)
        return labels_embedding

    def get_merge_embeddings(self, tree_stack):
        left_embeddings = []
        op_embeddings = []
        # on right tree => merge
        while len(tree_stack) >= 2 and tree_stack[-1].terminal and (not tree_stack[-2].terminal):
            left_embedding = tree_stack.pop().embedding
            op_embedding = tree_stack.pop().embedding
            left_embeddings.append(left_embedding)
            op_embeddings.append(op_embedding)
        return left_embeddings, op_embeddings
    
    # def fast_forward(self, class_embedding, tree_stacks, embed_node_index):
    def forward(self, class_embedding, tree_stacks, embed_node_index):
        # embed_node_index: batch_size
        batch_index = torch.arange(embed_node_index.size(0), device=embed_node_index.device)
        labels_embedding = class_embedding[batch_index, embed_node_index]
        merge_batch = []
        right_embeddings = []
        all_left_embeddings = []
        all_op_embeddings = []
        batch_step_size = []
        # get merge steps
        for batch_index, (node_label, tree_stack, label_embedding) in enumerate(zip(embed_node_index.cpu().tolist(), tree_stacks, labels_embedding)):
            # operations
            if node_label in self.op_set:
                tree_node = TreeEmbeddingNode(label_embedding, terminal=False)
                tree_stack.append(tree_node)    # no need to merge
            # numbers
            else:
                right_embedding = label_embedding
                left_embeddings, op_embeddings = self.get_merge_embeddings(tree_stack)
                current_step_size = len(left_embeddings)
                if current_step_size > 0:
                    merge_batch.append(batch_index)
                    right_embeddings.append(right_embedding)
                    all_left_embeddings.append(left_embeddings)
                    all_op_embeddings.append(op_embeddings)
                    batch_step_size.append(current_step_size)
                else:
                    tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
                    tree_stack.append(tree_node)    # no need to merge
        # data need to merge
        # batch parallel steps
        # sort all data to merge by step_count from high to low
        if len(merge_batch) > 0:
            embed_idx_size = list(enumerate(batch_step_size))
            embed_idx_size.sort(key=lambda idx_size: idx_size[1], reverse=True)
            embed_idx, batch_step_size = list(zip(*embed_idx_size))
            merge_batch = [merge_batch[idx] for idx in embed_idx]
            right_embeddings = [right_embeddings[idx] for idx in embed_idx]
            # convert batch_data to step_data
            # [batch1_embeddings, batch2_embeddings, ...]
            # batch1_embeddings: [step1_embedding, step2_embedding, ...]
            all_left_embeddings = [all_left_embeddings[idx] for idx in embed_idx]
            all_op_embeddings = [all_op_embeddings[idx] for idx in embed_idx]
            # [step1_embeddings, step2_embeddings, ...]
            # step1_embeddings: [batch1_embedding, batch2_embedding, ...]
            max_step_size = batch_step_size[0]
            # require batch_data in order by step_count from high to low
            serial_left_embeddings = [[batch_data[step_index] for batch_data in all_left_embeddings if step_index < len(batch_data)] for step_index in range(max_step_size)]
            serial_op_embeddings = [[batch_data[step_index] for batch_data in all_op_embeddings if step_index < len(batch_data)] for step_index in range(max_step_size)]
            step_batch_size = [len(batch_data) for batch_data in serial_left_embeddings]
            # batch merge
            right_embeddings = torch.stack(right_embeddings, dim=0)
            last_step_size = -1
            for size, left_embeddings, op_embeddings in zip(step_batch_size, serial_left_embeddings, serial_op_embeddings):
                # low step batch end merging, add merged embedding to tree_stack
                # require batch_data in order by step_count from high to low
                if last_step_size >= 0 and size != last_step_size:
                    end_size = last_step_size - size
                    for end_index in range(end_size):
                        end_index = size + end_index
                        batch_index = merge_batch[end_index]
                        right_embedding = right_embeddings[end_index]
                        tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
                        tree_stacks[batch_index].append(tree_node)
                last_step_size = size
                # high step batch continue merging
                right_embeddings = right_embeddings[:size]
                left_embeddings = torch.stack(left_embeddings, dim=0)
                op_embeddings = torch.stack(op_embeddings, dim=0)
                right_embeddings = self.merge(op_embeddings, left_embeddings, right_embeddings)
            # merged embedding for last step
            for end_index in range(last_step_size):
                batch_index = merge_batch[end_index]
                right_embedding = right_embeddings[end_index]
                tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
                tree_stacks[batch_index].append(tree_node)
        return labels_embedding 

class NodeEmbeddingNode:
    def __init__(self, node_hidden, node_context=None, label_embedding=None):
        self.node_hidden = node_hidden
        self.node_context = node_context
        self.label_embedding = label_embedding
        return

class DecomposeModel(nn.Module):
    def __init__(self, hidden_dim, dropout=0.4):
        super(DecomposeModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(p=dropout)
        self.l_decompose = GateNN(hidden_dim, hidden_dim*2, 0, dropout=dropout, one_layer=False)
        self.r_decompose = GateNN(hidden_dim, hidden_dim*2, hidden_dim, dropout=dropout, one_layer=False)
        return
    
    def serial_forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
    # def forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
        children_hidden = []
        finished_mask = []
        pad_hidden = torch.zeros(self.hidden_dim, device=labels_embedding.device)
        for node_stack, tree_stack, node_context, label_embedding in zip(node_stacks, tree_stacks, nodes_context, labels_embedding):
            # start from encoder_hidden
            # len == 0 => finished decode
            if len(node_stack) > 0:
                # left
                if not tree_stack[-1].terminal:
                    node_hidden = node_stack[-1].node_hidden    # parent, still need for right
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)   # add context and label of parent for right child
                    l_input = torch.cat((node_context, label_embedding), dim=-1)
                    l_input = self.dropout(l_input)
                    node_hidden = self.dropout(node_hidden)
                    child_hidden = self.l_decompose(node_hidden, l_input, None)
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for left child
                # right
                else:
                    node_stack.pop()    # left child or last node, no need
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()  # parent, no longer need
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding   # left tree
                        left_embedding = self.dropout(left_embedding)
                        r_input = torch.cat((node_context, label_embedding), dim=-1)
                        r_input = self.dropout(r_input)
                        node_hidden = self.dropout(node_hidden)
                        child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
                        node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for right child
                    # else finished decode
            # finished decode, pad
            if len(node_stack) == 0:
                child_hidden = pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
                finished_mask.append(1)
            else:
                finished_mask.append(0)
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        finished_mask = torch.tensor(finished_mask, device=children_hidden.device)
        return children_hidden, finished_mask
    
    # def fast_forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
    def forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
        hidden_flags = []
        left_batch_index = []
        left_node_hidden = []
        left_node_context = []
        left_label_embedding = []
        right_batch_index = []
        right_node_hidden = []
        right_node_context = []
        right_label_embedding = []
        right_left_embedding = []
        # batch left data and right data
        for batch_index, (node_stack, tree_stack, node_context, label_embedding) in enumerate(zip(node_stacks, tree_stacks, nodes_context, labels_embedding)):
            # start from encoder_hidden
            # len == 0 => finished decode
            if len(node_stack) > 0:
                # left
                if not tree_stack[-1].terminal:
                    node_hidden = node_stack[-1].node_hidden    # parent, still need for right
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)   # add context and label of parent for right child
                    left_batch_index.append(batch_index)
                    left_node_hidden.append(node_hidden)
                    left_node_context.append(node_context)
                    left_label_embedding.append(label_embedding)
                    index_in_left = len(left_batch_index) - 1
                    hidden_flags.append(('left', index_in_left))
                # right
                else:
                    node_stack.pop()    # left child or last node, no need
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()  # parent, no longer need
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding   # left tree
                        right_batch_index.append(batch_index)
                        right_node_hidden.append(node_hidden)
                        right_node_context.append(node_context)
                        right_label_embedding.append(label_embedding)
                        right_left_embedding.append(left_embedding)
                        index_in_right = len(right_batch_index) - 1
                        hidden_flags.append(('right', index_in_right))
                    else:
                        # finished decode
                        hidden_flags.append(('none',))   # pad
            else:
                hidden_flags.append(('none',))   # pad
        # batch left decompose
        if len(left_batch_index) > 0:
            node_hidden = torch.stack(left_node_hidden, dim=0)
            node_context = torch.stack(left_node_context, dim=0)
            label_embedding = torch.stack(left_label_embedding, dim=0)
            l_input = self.dropout(torch.cat((node_context, label_embedding), dim=-1))
            node_hidden = self.dropout(node_hidden)
            left_child_hidden = self.l_decompose(node_hidden, l_input, None)
        # batch right decompose
        if len(right_batch_index) > 0:
            node_hidden = torch.stack(right_node_hidden, dim=0)
            node_context = torch.stack(right_node_context, dim=0)
            label_embedding = torch.stack(right_label_embedding, dim=0)
            left_embedding = torch.stack(right_left_embedding, dim=0)
            left_embedding = self.dropout(left_embedding)
            r_input = self.dropout(torch.cat((node_context, label_embedding), dim=-1))
            node_hidden = self.dropout(node_hidden)
            right_child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
        # post process
        # for left
        for child_hidden_index, batch_index in enumerate(left_batch_index):
            child_hidden = left_child_hidden[child_hidden_index]
            node_stacks[batch_index].append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for left child
        # for right
        for child_hidden_index, batch_index in enumerate(right_batch_index):
            child_hidden = right_child_hidden[child_hidden_index]
            node_stacks[batch_index].append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for right child
        # for all after above
        children_hidden = []
        finished_mask = []
        pad_hidden = torch.zeros(self.hidden_dim, device=labels_embedding.device)
        for batch_index, node_stack, in enumerate(node_stacks):
            # finished decode, pad
            if len(node_stack) == 0:
                child_hidden = pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
                finished_mask.append(1)
            else:
                group, index_in_group = hidden_flags[batch_index]
                if group == "left":
                    child_hidden = left_child_hidden[index_in_group]
                elif group == "right":
                    child_hidden = right_child_hidden[index_in_group]
                finished_mask.append(0)
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        finished_mask = torch.tensor(finished_mask, device=children_hidden.device)
        return children_hidden, finished_mask

def copy_list(src_list):
    dst_list = [copy_list(item) if type(item) is list else item for item in src_list]
    return dst_list

class BeamNode:
    def __init__(self, score, nodes_hidden, finished_mask, node_stacks, tree_stacks, decoder_outputs_list, sequence_symbols_list):
        self.score = score
        self.nodes_hidden = nodes_hidden
        self.finished_mask = finished_mask
        self.node_stacks = node_stacks
        self.tree_stacks = tree_stacks
        self.decoder_outputs_list = decoder_outputs_list
        self.sequence_symbols_list = sequence_symbols_list
        return
    
    def copy(self):
        node = BeamNode(
            self.score,
            self.nodes_hidden,
            self.finished_mask,
            copy_list(self.node_stacks),
            copy_list(self.tree_stacks),
            copy_list(self.decoder_outputs_list),
            copy_list(self.sequence_symbols_list)
        )
        return node

class Decoder(nn.Module):
    def __init__(self, class_list, op_set, hidden_dim=512, max_decode_length=40, dropout=0.4):
        super(Decoder, self).__init__()
        class_size = len(class_list)
        self.max_decode_length = max_decode_length
        self.init_class_embed_order(class_list, op_set)
        self.embed_model = nn.Embedding(self.n_vocab, hidden_dim)
        self.predict = PredictModel(hidden_dim, class_size, self.symbol_vocab.size(0) - 1, dropout=dropout)
        op_idx_set = set(i for i, symbol in enumerate(class_list) if symbol in op_set)
        self.tree_embedding = TreeEmbeddingModel(hidden_dim, op_idx_set, dropout=dropout)
        self.decompose = DecomposeModel(hidden_dim, dropout=dropout)
        return

    def init_class_embed_order(self, class_list, op_set):
        # embed order: generator + pointer, with original order
        # used in predict_model, tree_embedding
        pointer_list = [token for token in class_list if 'temp_' in token]
        pad_list = [token for token in class_list if token == "pad"]
        op_list = [token for token in class_list if token not in pointer_list and token in op_set]
        symbol_list = pad_list + op_list
        const_list = [token for token in class_list if token not in pointer_list and token not in symbol_list]
        vocab_list = symbol_list + const_list
        embed_list = vocab_list + pointer_list
        vocab_dict = {token:i for i, token in enumerate(vocab_list)}
        self.n_vocab = len(vocab_dict)

        # pointer num index in class_list, for select only num pos from num_pos with op pos
        self.pointer_index = torch.LongTensor([class_list.index(token) for token in pointer_list])
        # generator symbol index in vocab, for generator symbol embedding
        self.symbol_vocab = torch.LongTensor([vocab_dict[token] for token in symbol_list])
        self.const_vocab = torch.LongTensor([vocab_dict[token] for token in const_list])
        # class_index -> embed_index, for tree embedding
        # embed order -> class order, for predict_model output
        self.class_to_embed_index = torch.LongTensor([embed_list.index(token) for token in class_list])
        self.on_device = False
        return
    
    def device_class_embed_order(self, any_input):
        device = any_input.device
        if not self.on_device:
            self.pointer_index = self.pointer_index.to(device)
            self.symbol_vocab = self.symbol_vocab.to(device)
            self.const_vocab = self.const_vocab.to(device)
            self.class_to_embed_index = self.class_to_embed_index.to(device)
            self.on_device = True
        return

    def get_pointer_num_pos(self, num_pos):
        batch_size = num_pos.size(0)
        pointer_num_pos = num_pos.index_select(dim=1, index=self.pointer_index)
        num_pos_occupied = pointer_num_pos.sum(dim=0) == -batch_size
        occupied_len = num_pos_occupied.size(-1)
        for i, elem in enumerate(reversed(num_pos_occupied.cpu().tolist())):
            if not elem:
                occupied_len = occupied_len - i
                break
        pointer_num_pos = pointer_num_pos[:, :occupied_len]
        return pointer_num_pos

    def get_pointer_embedding_mask(self, pointer_num_pos, encoder_outputs):
        # encoder_outputs: batch_size * seq_len * hidden_dim
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        batch_size, pointer_size = pointer_num_pos.size()
        batch_index = torch.arange(batch_size, device=pointer_num_pos.device)
        batch_index = batch_index.unsqueeze(1).expand(-1, pointer_size)
        # batch_size * pointer_len * hidden_dim
        pointer_embedding = encoder_outputs[batch_index, pointer_num_pos]
        # mask invalid pos -1
        pointer_embedding = pointer_embedding * (pointer_num_pos != -1).unsqueeze(-1)
        # subset of num_pos, invalid pos -1
        pointer_mask = pointer_num_pos != -1
        return pointer_embedding, pointer_mask
    
    def get_vocab_embedding_mask(self, batch_size):
        # generator_size * hidden_dim
        symbol_embedding = self.embed_model(self.symbol_vocab)
        const_embedding = self.embed_model(self.const_vocab)
        vocab_embedding = torch.cat((symbol_embedding, const_embedding), dim=0)
        # batch_size * generator_size * hidden_dim
        const_embedding = const_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        vocab_embedding = vocab_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        const_mask = torch.ones(batch_size, self.const_vocab.size(0), device=self.const_vocab.device).bool()
        return const_embedding, vocab_embedding, const_mask
    
    def get_class_embedding_mask(self, num_pos, encoder_outputs):
        # embedding: batch_size * size * hidden_dim
        # mask: batch_size * size
        const_embedding, vocab_embedding, const_mask = self.get_vocab_embedding_mask(num_pos.size(0))
        pointer_num_pos = self.get_pointer_num_pos(num_pos)
        pointer_embedding, pointer_mask = self.get_pointer_embedding_mask(pointer_num_pos, encoder_outputs)
        
        num_embedding = torch.cat((const_embedding, pointer_embedding), dim=1)
        num_mask =  torch.cat((const_mask, pointer_mask), dim=1)
        all_embedding = torch.cat((vocab_embedding, pointer_embedding), dim=1)
        return num_embedding, num_mask, all_embedding

    def init_stacks(self, encoder_hidden):
        batch_size = encoder_hidden.size(0)
        node_stacks = [[NodeEmbeddingNode(hidden, None, None)] for hidden in encoder_hidden]
        tree_stacks = [[] for _ in range(batch_size)]
        return node_stacks, tree_stacks

    def forward_step(self, node_stacks, tree_stacks, nodes_hidden, encoder_outputs, encoder_masks, class_embedding_masks, finished_mask, decoder_nodes_class=None):
        nodes_output, nodes_context = self.predict(nodes_hidden, encoder_outputs, encoder_masks, class_embedding_masks, finished_mask)
        # embed_index_order => class_order
        nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
        predict_nodes_class = nodes_output.topk(1)[1]
        # teacher
        if decoder_nodes_class is not None:
            nodes_class = decoder_nodes_class.view(-1)
        # no teacher
        else:
            nodes_class = predict_nodes_class.view(-1)
        # class_order => embed_index_order
        embed_nodes_index = self.class_to_embed_index[nodes_class]
        labels_embedding = self.tree_embedding(class_embedding_masks[-1], tree_stacks, embed_nodes_index)
        nodes_hidden, finished_mask = self.decompose(node_stacks, tree_stacks, nodes_context, labels_embedding)
        return nodes_output, predict_nodes_class, nodes_hidden, finished_mask
    
    def forward_teacher(self, decoder_nodes_label, decoder_init_hidden, encoder_outputs, encoder_masks, class_embedding_masks, max_length=None):
        decoder_outputs_list = []
        sequence_symbols_list = []
        batch_size = decoder_init_hidden.size(0)
        decoder_hidden = decoder_init_hidden
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        finished_mask = torch.zeros(batch_size, device=decoder_init_hidden.device).long()
        if decoder_nodes_label is not None:
            seq_len = decoder_nodes_label.size(1)
        else:
            seq_len = max_length
        for di in range(seq_len):
            if decoder_nodes_label is not None:
                decoder_node_class = decoder_nodes_label[:, di]
            else:
                decoder_node_class = None
            decoder_output, symbols, decoder_hidden, finished_mask = self.forward_step(node_stacks, tree_stacks, decoder_hidden, encoder_outputs, encoder_masks, class_embedding_masks, finished_mask, decoder_nodes_class=decoder_node_class)
            decoder_outputs_list.append(decoder_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, sequence_symbols_list

    def forward_beam(self, decoder_init_hidden, encoder_outputs, encoder_masks, class_embedding_masks, max_length, beam_width=1):
        # only support batch_size == 1
        batch_size = decoder_init_hidden.size(0)
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        finished_mask = torch.zeros(batch_size, device=decoder_init_hidden.device).long()
        beams = [BeamNode(0, decoder_init_hidden, finished_mask, node_stacks, tree_stacks, [], [])]
        for _ in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                # finished stack-guided decoding
                if len(b.node_stacks) == 0:
                    current_beams.append(b)
                    continue
                # unfinished decoding
                # batch_size == 1
                # batch_size * class_size
                nodes_output, nodes_context = self.predict(b.nodes_hidden, encoder_outputs, encoder_masks, class_embedding_masks, b.finished_mask)
                # embed_index_order => class_order
                nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
                # batch_size * beam_width
                top_value, top_index = nodes_output.topk(beam_width)
                top_value = torch.exp(top_value)
                for predict_score, predicted_symbol in zip(top_value.split(1, dim=-1), top_index.split(1, dim=-1)):
                    nb = b.copy()
                    # class_order => embed_index_order
                    embed_nodes_index = self.class_to_embed_index[predicted_symbol.view(-1)]
                    labels_embedding = self.tree_embedding(class_embedding_masks[-1], nb.tree_stacks, embed_nodes_index)
                    nodes_hidden, finished_mask = self.decompose(nb.node_stacks, nb.tree_stacks, nodes_context, labels_embedding, pad_node=False)

                    nb.score = b.score + predict_score.item()
                    nb.nodes_hidden = nodes_hidden
                    nb.finished_mask = finished_mask
                    nb.decoder_outputs_list.append(nodes_output)
                    nb.sequence_symbols_list.append(predicted_symbol)
                    current_beams.append(nb)
            beams = sorted(current_beams, key=lambda b:b.score, reverse=True)
            beams = beams[:beam_width]
            all_finished = True
            for b in beams:
                if len(b.node_stacks[0]) != 0:
                    all_finished = False
                    break
            if all_finished:
                break
        output = beams[0]
        return output.decoder_outputs_list, output.sequence_symbols_list

    def forward(self, encoder_hidden, encoder_outputs, encoder_masks, num_pos, targets=None, beam_width=None):
        self.device_class_embed_order(num_pos)
        class_embedding_masks = self.get_class_embedding_mask(num_pos, encoder_outputs)
        decoder_init_hidden = encoder_hidden

        if targets is not None:
            max_length = targets.size(1)
        else:
            max_length = self.max_decode_length
        
        if beam_width is not None:
            return self.forward_beam(decoder_init_hidden, encoder_outputs, encoder_masks, class_embedding_masks, max_length, beam_width)
        else:
            return self.forward_teacher(targets, decoder_init_hidden, encoder_outputs, encoder_masks, class_embedding_masks, max_length)
