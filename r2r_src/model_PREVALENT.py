import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args

from vlnbert.vlnbert_init import get_vlnbert_models

class myKL(nn.Module):
    def __init__(self):
        super(myKL, self).__init__()
        self.disloss = nn.KLDivLoss()
    def forward(self, text_embeds, text_embeds_1):
        log_probs1 = F.log_softmax(text_embeds_1.clone(), 1)
        probs2 = F.softmax(text_embeds.clone().detach(), 1)
        dis_loss = self.disloss(log_probs1, probs2)

        log_probs2 = F.log_softmax(text_embeds.clone(), 1)
        probs1 = F.softmax(text_embeds_1.clone().detach(), 1)
        dis_loss += self.disloss(log_probs2, probs1)
        return dis_loss

class VLNBERT(nn.Module):
    def __init__(self, feature_size=2048+128):
        super(VLNBERT, self).__init__()
        print('\nInitalizing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.vln_bert.config.directions = 4  # a preset random number

        hidden_size = self.vln_bert.config.hidden_size
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.outproj = nn.Linear(hidden_size, 2048)
        self.acti = nn.Tanh()
        self.visproj = nn.Linear(2048, 1)
        self.disloss = myKL()

    def forward(self, mode, sentence, token_type_ids=None,
                attention_mask=None, lang_mask=None, vis_mask=None,
                position_ids=None, action_feats=None, pano_feats=None, cand_feats=None, t=0, seq_lengths=None, att_drop_rate=None):

        if mode == 'language':
            text_embeds = self.vln_bert(mode, sentence, attention_mask=attention_mask, lang_mask=lang_mask,)
            return text_embeds #init_state, encoded_sentence

        elif mode == 'visual':
            cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
            
            h_t, lang_list, visn_list = self.vln_bert(mode, sentence,
                attention_mask=attention_mask, lang_mask=lang_mask, vis_mask=vis_mask, img_feats=cand_feats,t=t, seq_lengths=seq_lengths)
            # vis output as classification
            pred_vis = self.outproj(visn_list[-1][:, t:, :])
            pred_vis = self.acti(pred_vis)
            logit = self.visproj(pred_vis).squeeze(2)

            lang_mask_1 = torch.rand_like(lang_mask.float()) < (1-att_drop_rate)
            lang_mask_1 = lang_mask_1.long()
            lang_mask_1 = lang_mask_1.mul(lang_mask)
            h_t_1, lang_list_1, visn_list_1 = self.vln_bert(mode, sentence,
                attention_mask=attention_mask, lang_mask=lang_mask_1, vis_mask=vis_mask, img_feats=cand_feats,t=t, seq_lengths=seq_lengths)  
            
            dis_loss = 0.
            for lang_output, visn_output, lang_output_1, visn_output_1 in lang_list, visn_list, lang_list_1, visn_list_1:
                dis_loss += self.disloss(lang_output, lang_output_1)
                dis_loss += self.disloss(visn_output, visn_output_1)
            return h_t, pred_vis, logit, dis_loss

        else:
            ModuleNotFoundError


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
