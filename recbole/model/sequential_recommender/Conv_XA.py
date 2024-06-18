# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 22:02
# @Author  : Eunjo Jang
# @Email   : wkddmswh99@sookmyung.ac.kr

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer,CATTransformerEncoder
from recbole.model.loss import BPRLoss
import copy




class Conv_XA(SequentialRecommender):

    def __init__(self, config, dataset):
        super(Conv_XA, self).__init__(config, dataset)

        # Load parameters from config
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['fusion_type']
        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']

        # Define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # Feature embedding layers
        self.feature_embed_layer_list = nn.ModuleList([
            copy.deepcopy(FeatureSeqEmbLayer(dataset, self.attribute_hidden_size[i], [self.selected_features[i]], self.pooling_mode, self.device))
            for i in range(self.num_feature_field)
        ])

        # Transformer encoder
        self.trm_encoder = CATTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.selected_features),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

        # Attribute information
        self.n_attributes = {attribute: len(dataset.field2token_id[attribute]) for attribute in self.selected_features}
        self.cate_n_attributes = len(dataset.field2token_id[self.selected_features[0]])  # Assuming the first attribute is categorical
            
        # Attribute predictor
        if self.attribute_predictor == 'MLP':
            self.ap = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.cate_n_attributes)
            )
        elif self.attribute_predictor == 'linear':
            self.ap = nn.ModuleList([
                copy.deepcopy(nn.Linear(self.hidden_size, self.n_attributes[attribute]))
                for attribute in self.selected_features
            ])


        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        # Loss functions
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Unsupported loss_type. Choose from ['BPR', 'CE'].")


        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
            

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        
        # Embed item sequences
        item_emb = self.item_embedding(item_seq)  # shape: (batch_size, seq_length, hidden_size)  
              
        # Position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embed = self.position_embedding(position_ids)  # shape: (batch_size, seq_length, hidden_size)


        # Feature embedding
        feature_table = []
        for feature_embed_layer in self.feature_embed_layer_list:
            sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            sparse_embedding = sparse_embedding['item']
            dense_embedding = dense_embedding['item']
            if sparse_embedding is not None:
                feature_table.append(sparse_embedding)
            if dense_embedding is not None:
                feature_table.append(dense_embedding)

            
        # Assuming the first feature is categorical
        cate_emb = feature_table[0]

        # Add item and position embeddings
        input_emb = item_emb + position_embed
        input_emb = self.LayerNorm(input_emb)  # Layer normalization
        input_emb = self.dropout(input_emb)    # Dropout for regularization

        # Normalize and apply dropout to category embeddings
        cate_emb = self.LayerNorm(cate_emb)
        cate_emb = self.dropout(cate_emb)
        
        # Transformer encoder
        output = self.trm_encoder(input_emb, cate_emb, output_all_encoded_layers=False)
        item_hidden_states, cate_hidden_states = output[0]

        # Fuse item and category hidden states
        fused_output = item_hidden_states + cate_hidden_states  # shape: (batch_size, seq_length, hidden_size)
        
        # Gather the final sequence output
        seq_output = self.gather_indexes(fused_output, item_seq_len - 1)  # shape: (batch_size, hidden_size)

        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]  # shape: (batch_size, seq_length)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # shape: (batch_size)
        
        # Forward pass
        seq_output = self.forward(item_seq, item_seq_len)  # shape: (batch_size, hidden_size)
        
        pos_items = interaction[self.POS_ITEM_ID]  # Positive item IDs for each user

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # shape: (batch_size)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # shape: (batch_size)
            loss = self.loss_fct(pos_score, neg_score)
            return loss

        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight  # shape: (num_items, hidden_size)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # shape: (batch_size, num_items)
            loss = self.loss_fct(logits, pos_items)

            # If attribute predictor is used
            if self.attribute_predictor not in ['', 'not']:
                loss_dic = {'item_loss': loss}
                attribute_loss_sum = 0

                for i, a_predictor in enumerate(self.ap):
                    attribute_logits = a_predictor(seq_output)
                    attribute_labels = interaction[self.selected_features[i]]
                    attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes[self.selected_features[i]])

                    if len(attribute_labels.shape) > 2:
                        attribute_labels = attribute_labels.sum(dim=1)
                    attribute_labels = attribute_labels.float()
                    attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
                    attribute_loss = torch.mean(attribute_loss[:, 1:])
                    loss_dic[self.selected_features[i]] = attribute_loss

                if self.num_feature_field == 1:
                    total_loss = loss + self.lamdas[0] * attribute_loss
                else:
                    for i, attribute in enumerate(self.selected_features):
                        attribute_loss_sum += self.lamdas[i] * loss_dic[attribute]
                    total_loss = loss + attribute_loss_sum
                    loss_dic['total_loss'] = total_loss
            else:
                total_loss = loss

            return total_loss

        else:
            raise NotImplementedError("Unsupported loss_type. Choose from ['BPR', 'CE'].")


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output= self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores
