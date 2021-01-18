import numpy
import torch
import sklearn
from fastai.text.all import *
def map_embs(model,dl,pre_trained_dict):
    model.emb.state_dict()['weight']
    for i in range(len(dl.numericalize.vocab)):
        try:
            model.emb.state_dict()['weight'][i] = torch.tensor(pre_trained_dict[dl.numericalize.vocab[i]]);
        except:
            print(dl.numericalize.vocab[i])
            model.emb.state_dict()['weight'][i] = torch.tensor(pre_trained_dict['xxunk']);
    

class Pad_Input(ItemTransform):
    def __init__(self,length):
        self.fixed_length = length
        print('initialized')
    def encodes(self,samples, pad_idx=1, pad_fields=0, pad_first=False, backwards=False):
        "Function that collect `samples` and adds padding"
        
        self.pad_idx = pad_idx
        pad_fields = L(pad_fields)
        max_len_l = [self.fixed_length]
        #pad_fields.map(lambda f: max([len(s[f]) for s in samples]))
        if backwards: pad_first = not pad_first
        def _f(field_idx, x):
            if field_idx not in pad_fields: return x
            idx = pad_fields.items.index(field_idx) #TODO: remove items if L.index is fixed
            sl = slice(-len(x), sys.maxsize) if pad_first else slice(0, len(x))
            pad =  x.new_zeros(max_len_l[idx]-x.shape[0])+pad_idx
            x1 = torch.cat([pad, x] if pad_first else [x, pad])
            if backwards: x1 = x1.flip(0)
            return retain_type(x1, x)
        return [tuple(map(lambda idxx: _f(*idxx), enumerate(s))) for s in samples]
    def decodes(self, o:TensorText):
        pad_idx = self.pad_idx if hasattr(self,'pad_idx') else 1
        return o[o != pad_idx]


def acc(inp,target):   
    return sklearn.metrics.accuracy_score(target.cpu().numpy()==1,inp.cpu().numpy()>0.5)

def recall(inp,target):   
    return sklearn.metrics.recall_score(target.cpu().numpy()==1,inp.cpu().numpy()>0.5)

def precision(inp,target):
    return sklearn.metrics.precision_score(target.cpu().numpy()==1,inp.cpu().numpy()>0.5)

def f1(inp,target):
    return sklearn.metrics.f1_score(target.cpu().numpy()==1,inp.cpu().numpy()>0.5)

def bce(inp,target):
    target = target.type(torch.float)
    return nn.functional.binary_cross_entropy(inp,target)


