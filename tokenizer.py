from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer
from preprocessing import *
from fastai.text.all import *
import gc
class CustomTokenizer:
    '''BertTok is a class created to pass the Bert tokenizer to fastai library API '''
    def __init__(self,tokenizer):
        self.tok = tokenizer
    def __call__(self,t):
        # It maps its input to string type and enables iteration over the input
        # It is needed for passing the tokenizer to the fastai API
        it = map(str,t)
        t = [self.tok.tokenize(i) for i in it]
        return L(t)

def get_tokenizer(model_name , tokenizer_link, keep_emoji=True,special_prep=True):
    
    
    rules = [rep_hash,rep_link,rep_emojis]+defaults.text_proc_rules
    if special_prep:
        arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=keep_emoji)
        rules.append(arabert_prep.preprocess)
    special_tokens = defaults.text_spec_tok+ ['xxhash','xxemoji']

    
    # Download the Custom Tokenizer
    custom_tokenizer = AutoTokenizer.from_pretrained(tokenizer_link)

    #It replace the special tokens of the tokenizer with our special token. e.g, pad_token : '[PAD]' --> 'xxpad' 
    custom_tokenizer.add_special_tokens({'bos_token':'xxbos', 'unk_token':'xxunk' , 'pad_token':'xxpad','eos_token':'xxeos'})
    #Adding the special tokens to Bert tokenizer, It should add 9 tokens
    [custom_tokenizer.add_tokens(i,special_tokens=True) for i in special_tokens[4:]]

    tkn = CustomTokenizer(custom_tokenizer)
    return tkn,rules


def get_dl(path,tokenizer=None,rules=None,folders=['pos','neg'],skip_if_exists=False):
    get_tweets = partial(get_files,folders=folders)

    sorted_vocab = {k: v for k, v in sorted(tokenizer.tok.vocab.items(), key=lambda item: item[1])}
    pad_input = Pad_Input()
    return DataBlock(
    blocks=(TextBlock.from_folder(path,tok=tokenizer,rules=rules,vocab=list(sorted_vocab.keys())
                                  ,skip_if_exists=skip_if_exists),CategoryBlock),
        get_y=parent_label,get_items=get_tweets, splitter=RandomSplitter(0.2)
    ).dataloaders(path, path=path, bs=8,before_batch=partial(pad_input, pad_idx=tokenizer.tok.pad_token_id) )






