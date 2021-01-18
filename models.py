from fastai.text.all import *
from transformers import  AutoModelForSequenceClassification
from metrics import *
from tokenizer import *
import gc
class Custom_Transformer(nn.Module):
    def __init__(self,len_tokenizer,path,**kwargs):
        super().__init__()
        # Get pre-trained arabert model with single layer of single output
        self.model = AutoModelForSequenceClassification.from_pretrained(path,output_attentions=False,num_labels=1, **kwargs)
        # Resize the mebedding due to the added tokens
        self.model.resize_token_embeddings(len_tokenizer)
    def forward(self,x):
        x=self.model(x)
        x = torch.sigmoid(x[0].view(-1))
        return x

def get_learner(dl,loss,custom_tokenizer,pre_trained_config_path,**kwargs):
    model = Custom_Transformer(len(custom_tokenizer.tok),pre_trained_config_path,**kwargs)

    learn = Learner(dl,model,loss_func=loss,opt_func=Adam,metrics=[f1,precision,recall,acc])

    #Freeze all layers except normalization layer and classifier layers
    learn.freeze_to(1)  
    [i.requires_grad_(False) for i in list(learn.model.model.parameters()) ]
    layer_unfreeze(list(learn.model.model.children())[-1]) 
    

    lr_min,lr_steep = learn.lr_find(); lr_min,lr_steep
    plt.pause(0.001)
    return learn

def quick_run(learn,lr,lr_max):
    #train for 2 epochs
    learn.lr = lr
    learn.fit_one_cycle(2,lr_max = lr_max)

    learn.unfreeze()

def quick_run(learn):
    lr = input('lr = ?')
    lr_max = input('maximum lr = ? ')
    #train for 2 epochs
    learn.lr = float(lr)
    learn.fit_one_cycle(2,lr_max = float(lr_max))

    learn.unfreeze()

    #Half the learning rate and train for 4 epochs
    learn.lr = learn.lr /2
    learn.fit_one_cycle(4,lr_max = learn.lr)

def layer_unfreeze(obj):
    [i.requires_grad_(True) for i in obj.parameters()]


def release_memory(learn):
    learn.model = None
    learn.opt = None
    gc.collect()
    torch.cuda.empty_cache()


def full_stack(model_name,pre_trained_config_path,dataset,data_path,store_path,**kwargs):
    #Get tokenizer
    custom_tokenizer,rules = get_tokenizer(model_name,pre_trained_config_path)

    #### Dataset & Dataloaders

    dls_lm = get_dl(data_path,custom_tokenizer,rules)

    dls_lm.show_batch(max_n=2)

    # Model

    ### Traning :sentiment

    #pad_token_id=custom_tokenizer.tok.pad_token_id (must be used for ara-gpt2)

    learn = get_learner(dls_lm,bce,custom_tokenizer,pre_trained_config_path,**kwargs)

    #[(j,i.requires_grad) for j,i in list(learn.model.model.named_parameters()) ]

    quick_run(learn)

    #torch.save(learn.model.state_dict(),f'{store_path}/{model_name}_{dataset}')
    #learn.model.load_state_dict(torch.load(f'{store_path}{model_name}'))

    release_memory(learn)
