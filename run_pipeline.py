import stanza
import sys
from stanza.utils.conll import CoNLL
#  from stanza.models.common.pretrain import Pretrain
# pt = Pretrain("new.pt","/home/django/hywshtem/tokenization/stanzaa/hy/word2vec/Armenian/hy.pretrain.pt")
# pt.load()
east = stanza.Pipeline(lang = 'hy',
                       processors ='tokenize, mwt, lemma, pos, depparse',
                       tokenize_model_path='saved_models/tokenize/hy_armtdp_tokenizer.pt',
                       mwt_model_path='saved_models/mwt/hy_armtdp_mwt_expander.pt',
                       lemma_model_path='saved_models/lemma/hy_armtdp_lemmatizer.pt',
                    #    pos_pretrain_path='/home/django/hywshtem/tokenization/stanzaa/hy/word2vec/Armenian/hy.pretrain.pt',
                       pos_model_path='saved_models/pos/hy_armtdp_tagger.pt',
                    #    depparse_pretrain_path='/home/django/hywshtem/tokenization/stanzaa/hy/word2vec/Armenian/hy.pretrain.pt',
                       depparse_model_path='saved_models/depparse/hy_armtdp_parser.pt')

west = stanza.Pipeline(lang = 'hy',
                       processors ='tokenize, mwt, lemma, pos, depparse',
                       pos_model_path='saved_models/pos/hy_armtdp_tagger_hyw.pt',
                       tokenize_model_path='saved_models/tokenize/hy_armtdp_tokenizer_hyw.pt',
                       depparse_model_path='saved_models/depparse/hy_armtdp_parser_hyw.pt',
                       lemma_model_path='saved_models/lemma/hy_armtdp_lemmatizer_hyw.pt',
                       mwt_model_path='saved_models/mwt/hy_armtdp_mwt_expander_hyw.pt')
hy_hyw = stanza.Pipeline(lang = 'hy',
                       processors ='tokenize, mwt, lemma, pos, depparse',
                       pos_model_path='saved_models/pos/hy_armtdp_tagger_hy_hyw.pt',
                       tokenize_model_path='saved_models/tokenize/hy_armtdp_tokenizer_hy_hyw.pt',
                       depparse_model_path='saved_models/depparse/hy_armtdp_parser_hy_hyw.pt',
                       lemma_model_path='saved_models/lemma/hy_armtdp_lemmatizer_hy_hyw.pt',
                       mwt_model_path='saved_models/mwt/hy_armtdp_mwt_expander_hy_hyw.pt')
# text = sys.argv[1]
def run_east(text):
    # print(type(text))
    doc = east(text)
    conll_output = CoNLL.dict2conll(doc.to_dict(), 'work.conllu')
    print(conll_output)
    return conll_output
# run_east(text)

def run_west(text):
    # print(type(text))
    doc = west(text)
    conll_output = CoNLL.dict2conll(doc.to_dict(), 'work.conllu')
    return conll_output


def run_hy_hyw(text):
    doc = hy_hyw(text)
    conll_output = CoNLL.dict2conll(doc.to_dict(), 'work.conllu')
    return conll_output