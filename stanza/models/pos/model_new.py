import numpy as np
import stanza.models.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from stanza.models.common.biaffine import BiaffineScorer
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.dropout import WordDropout
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common.char_model import CharacterModel


class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if not share_hid:
            # upos embeddings
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb',
                               nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        # recurrent layers
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True,
                                      bidirectional=True, dropout=self.args['dropout'],
                                      rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        if share_hid:
            clf_constructor = lambda insize, outsize: nn.Linear(insize, outsize)
        else:
            self.xpos_hid = nn.Linear(self.args['hidden_dim'] * 2,
                                      self.args['deep_biaff_hidden_dim'] if not isinstance(vocab['xpos'],
                                                                                           CompositeVocab) else
                                      self.args['composite_deep_biaff_hidden_dim'])
            self.ufeats_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['composite_deep_biaff_hidden_dim'])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args['tag_emb_dim'], outsize)

        if isinstance(vocab['xpos'], CompositeVocab):
            self.xpos_clf = nn.ModuleList()
            for l in vocab['xpos'].lens():
                self.xpos_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))
        else:
            self.xpos_clf = clf_constructor(self.args['deep_biaff_hidden_dim'], len(vocab['xpos']))
            if share_hid:
                self.xpos_clf.weight.data.zero_()
                self.xpos_clf.bias.data.zero_()

        self.ufeats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            if share_hid:
                self.ufeats_clf.append(clf_constructor(self.args['deep_biaff_hidden_dim'], l))
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()
            else:
                self.ufeats_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx,
                sentlens, wordlens):

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(
        self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(),
        self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data
        # print('sentlens', sentlens)
        # print('len_sent_lens', len(sentlens))

        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        preds = [pad(upos_pred).max(2)[1]]

        # print('upos_pred', upos_pred)
        # print('softmax_upos', pad(F.softmax(upos_pred, dim=1)))
        # print('pad upos preds', pad(upos_pred))
        softmax_upos = pad(F.softmax(upos_pred, dim=1))
        softmax_scores = softmax_upos.max(2, keepdim=True)[0]

        # print('softmax_upos', softmax_upos)
        # print('softmax_upos max', softmax_scores)
        # print('softmax_upos max index', softmax_upos.max(2, keepdim=True)[1])
        # print('upos pred', preds)
        # print('softmax_upos shape', softmax_upos.shape)

        upos = pack(upos).data
        # print('upos', upos)
        # print('upos.view(-1)', upos.view(-1))
        # print('upos_pred.view(-1, upos_pred.size(-1))', upos_pred.view(-1, upos_pred.size(-1)))
        loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))

        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid

            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))

            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(upos_pred.max(1)[1])

            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))

        xpos = pack(xpos).data
        if isinstance(self.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab['xpos'])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, i].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(pad(xpos_pred).max(2)[1])

        ufeats_preds = []
        ufeats_preds_score = []  # liana
        softmax_vals = []  # liana
        ufeats = pack(ufeats).data
        # print('ufeats_hid', ufeats_hid) # same as upos_hid
        # print('feats',self.vocab['feats'])
        # print('len feats',len(self.vocab['feats']))
        for i in range(len(self.vocab['feats'])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            # print(i)
            # print('ufeats_pred', ufeats_pred) #tadaam
            ufeats_preds_score.append(pad(ufeats_pred))
            softmax_val = pad(F.softmax(ufeats_pred, dim=1))
            softmax_vals.append(softmax_val.max(2, keepdim=True)[0])
            # print('pad ufeats pred', pad(ufeats_pred)) this
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
            # print("softmax_vals", softmax_val)
            # print('softmax argmax', softmax_val.max(2, keepdim=True)[0])
            # print('ufeats pred', pad(ufeats_pred).max(2, keepdim=True)[1])
            # print('max softmax vals', softmax_val.max(2, keepdim=True)[1])
        # print('ufeats pred', ufeats_preds)
        # print('len ufeats_preds', len(ufeats_preds))
        # print('ufeats_pred_score', ufeats_preds_score)
        # print('Softmax ufeats_preds_score',softmax_vals)

        ufeats_scores = []
        softmax_ufeats_scores = []
        ufeats_scores.append(torch.cat(ufeats_preds_score, 2))
        softmax_ufeats_scores.append(torch.cat(softmax_vals, 2))

        preds.append(torch.cat(ufeats_preds, 2))

        # here
        # print('ufeats preds', torch.cat(ufeats_preds, 2))
        # print('soft', softmax_ufeats_scores)
        # print(preds[2].shape)
        # print(softmax_ufeats_scores[0].shape)

        sent_lens = torch.tensor(sentlens).reshape(len(sentlens), 1)
        sent_pos_scores = softmax_scores.sum(1) / sent_lens.to('cuda:0')
        # print("sent_pos_scores", sent_pos_scores)
        sent_feats_scores = softmax_ufeats_scores[0].max(2, keepdim=True)[0].sum(1) / sent_lens.to('cuda:0')
        # print("sent_feats_scores", sent_feats_scores)
        score = (sent_feats_scores + sent_pos_scores) / 2
        config.scores.append(score.tolist())
        with open('stanza/final_scores_similar.csv', 'a') as f:
            for i in score.tolist():
                f.write(str(i[0]))
                f.write('\n')
        with open('stanza/score_similar.csv', 'a') as f:
            f.write(str(score))
            f.write('\n')
        # print('to_list', config.scores[0].tolist())
        # print('verjy', config.scores)

        # print('preds in model', preds)
        # print('ufeats preds', ufeats_preds)
        # print('len ufeats_preds', len(ufeats_preds))
        # print('Len Softmax ufeats_preds_score',len(softmax_vals))
        # print('ban', ban)
        # print('ban_Soft', ban_soft)
        # print('ban[0]', type(ban[0]))

        return loss, preds

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from stanza.models.common.biaffine import BiaffineScorer
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.dropout import WordDropout
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common.char_model import CharacterModel


class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if not share_hid:
            # upos embeddings
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb',
                               nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        # recurrent layers
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True,
                                      bidirectional=True, dropout=self.args['dropout'],
                                      rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        if share_hid:
            clf_constructor = lambda insize, outsize: nn.Linear(insize, outsize)
        else:
            self.xpos_hid = nn.Linear(self.args['hidden_dim'] * 2,
                                      self.args['deep_biaff_hidden_dim'] if not isinstance(vocab['xpos'],
                                                                                           CompositeVocab) else
                                      self.args['composite_deep_biaff_hidden_dim'])
            self.ufeats_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['composite_deep_biaff_hidden_dim'])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args['tag_emb_dim'], outsize)

        if isinstance(vocab['xpos'], CompositeVocab):
            self.xpos_clf = nn.ModuleList()
            for l in vocab['xpos'].lens():
                self.xpos_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))
        else:
            self.xpos_clf = clf_constructor(self.args['deep_biaff_hidden_dim'], len(vocab['xpos']))
            if share_hid:
                self.xpos_clf.weight.data.zero_()
                self.xpos_clf.bias.data.zero_()

        self.ufeats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            if share_hid:
                self.ufeats_clf.append(clf_constructor(self.args['deep_biaff_hidden_dim'], l))
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()
            else:
                self.ufeats_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx,
                sentlens, wordlens):

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(
        self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(),
        self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data

        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        preds = [pad(upos_pred).max(2)[1]]

        upos = pack(upos).data
        loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))

        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid

            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))

            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(upos_pred.max(1)[1])

            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))

        xpos = pack(xpos).data
        if isinstance(self.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab['xpos'])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, i].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(pad(xpos_pred).max(2)[1])

        ufeats_preds = []
        ufeats = pack(ufeats).data
        for i in range(len(self.vocab['feats'])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
        preds.append(torch.cat(ufeats_preds, 2))

        return loss, preds

import numpy as np
import stanza.models.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from stanza.models.common.biaffine import BiaffineScorer
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.dropout import WordDropout
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common.char_model import CharacterModel


class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if not share_hid:
            # upos embeddings
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb',
                               nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        # recurrent layers
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True,
                                      bidirectional=True, dropout=self.args['dropout'],
                                      rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        if share_hid:
            clf_constructor = lambda insize, outsize: nn.Linear(insize, outsize)
        else:
            self.xpos_hid = nn.Linear(self.args['hidden_dim'] * 2,
                                      self.args['deep_biaff_hidden_dim'] if not isinstance(vocab['xpos'],
                                                                                           CompositeVocab) else
                                      self.args['composite_deep_biaff_hidden_dim'])
            self.ufeats_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['composite_deep_biaff_hidden_dim'])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args['tag_emb_dim'], outsize)

        if isinstance(vocab['xpos'], CompositeVocab):
            self.xpos_clf = nn.ModuleList()
            for l in vocab['xpos'].lens():
                self.xpos_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))
        else:
            self.xpos_clf = clf_constructor(self.args['deep_biaff_hidden_dim'], len(vocab['xpos']))
            if share_hid:
                self.xpos_clf.weight.data.zero_()
                self.xpos_clf.bias.data.zero_()

        self.ufeats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            if share_hid:
                self.ufeats_clf.append(clf_constructor(self.args['deep_biaff_hidden_dim'], l))
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()
            else:
                self.ufeats_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx,
                sentlens, wordlens):

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(
        self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(),
        self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data
        # print('sentlens', sentlens)
        # print('len_sent_lens', len(sentlens))

        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        preds = [pad(upos_pred).max(2)[1]]

        # print('upos_pred', upos_pred)
        # print('softmax_upos', pad(F.softmax(upos_pred, dim=1)))
        # print('pad upos preds', pad(upos_pred))
        softmax_upos = pad(F.softmax(upos_pred, dim=1))
        softmax_scores = softmax_upos.max(2, keepdim=True)[0]

        # print('softmax_upos', softmax_upos)
        # print('softmax_upos max', softmax_scores)
        # print('softmax_upos max index', softmax_upos.max(2, keepdim=True)[1])
        # print('upos pred', preds)
        # print('softmax_upos shape', softmax_upos.shape)

        upos = pack(upos).data
        # print('upos', upos)
        # print('upos.view(-1)', upos.view(-1))
        # print('upos_pred.view(-1, upos_pred.size(-1))', upos_pred.view(-1, upos_pred.size(-1)))
        loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))

        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid

            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))

            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(upos_pred.max(1)[1])

            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))

        xpos = pack(xpos).data
        if isinstance(self.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab['xpos'])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, i].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(pad(xpos_pred).max(2)[1])

        ufeats_preds = []
        ufeats_preds_score = []  # liana
        softmax_vals = []  # liana
        ufeats = pack(ufeats).data
        # print('ufeats_hid', ufeats_hid) # same as upos_hid
        # print('feats',self.vocab['feats'])
        # print('len feats',len(self.vocab['feats']))
        for i in range(len(self.vocab['feats'])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            # print(i)
            # print('ufeats_pred', ufeats_pred) #tadaam
            ufeats_preds_score.append(pad(ufeats_pred))
            softmax_val = pad(F.softmax(ufeats_pred, dim=1))
            softmax_vals.append(softmax_val.max(2, keepdim=True)[0])
            # print('pad ufeats pred', pad(ufeats_pred)) this
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
            # print("softmax_vals", softmax_val)
            # print('softmax argmax', softmax_val.max(2, keepdim=True)[0])
            # print('ufeats pred', pad(ufeats_pred).max(2, keepdim=True)[1])
            # print('max softmax vals', softmax_val.max(2, keepdim=True)[1])
        # print('ufeats pred', ufeats_preds)
        # print('len ufeats_preds', len(ufeats_preds))
        # print('ufeats_pred_score', ufeats_preds_score)
        # print('Softmax ufeats_preds_score',softmax_vals)

        ufeats_scores = []
        softmax_ufeats_scores = []
        ufeats_scores.append(torch.cat(ufeats_preds_score, 2))
        softmax_ufeats_scores.append(torch.cat(softmax_vals, 2))

        preds.append(torch.cat(ufeats_preds, 2))

        # here
        # print('ufeats preds', torch.cat(ufeats_preds, 2))
        # print('soft', softmax_ufeats_scores)
        # print(preds[2].shape)
        # print(softmax_ufeats_scores[0].shape)

        sent_lens = torch.tensor(sentlens).reshape(len(sentlens), 1)
        sent_pos_scores = softmax_scores.sum(1) / sent_lens.to('cuda:0')
        # print("sent_pos_scores", sent_pos_scores)
        sent_feats_scores = (softmax_ufeats_scores[0].max(2, keepdim=True)[0].sum(1)) / sent_lens.to('cuda:0')
        # print("sent_feats_scores", sent_feats_scores)
        score = (sent_feats_scores + sent_pos_scores) / 2

        config.scores.append(sent_pos_scores.tolist())
        with open('stanza/pos_similar_score.csv', 'a') as f:
            for i in sent_pos_scores.tolist():
                f.write(str(i[0]))
                f.write('\n')
        with open('stanza/feats_similar_score.csv', 'a') as f:
            for i in sent_feats_scores.tolist():
                f.write(str(i[0]))
                f.write('\n')

        # with open('stanza/scores.csv', 'a') as f:
        #     f.write(str(score))
        #     f.write('\n')
        # print('to_list', config.scores[0].tolist())
        # print('verjy', config.scores)

        # print('preds in model', preds)
        # print('ufeats preds', ufeats_preds)
        # print('len ufeats_preds', len(ufeats_preds))
        # print('Len Softmax ufeats_preds_score',len(softmax_vals))
        # print('ban', ban)
        # print('ban_Soft', ban_soft)
        # print('ban[0]', type(ban[0]))

        return loss, preds




