import pandas as pd
import os
import json

import elements
import nltk
import demoji

arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))

# Pre-procssing Methods

def rep_https(t):

    t = t.split()
    url_idx = [i for i,x in enumerate(t) if 'https' in x] 
    for i in url_idx: t[i] = 'xxurl'
    return ' '.join(t) 

def rep_mention(t):
    t = t.split()
    url_idx = [i for i,x in enumerate(t) if '@' in x] 
    for i in url_idx: t[i] = 'xxmention'
    
    return ' '.join(t) 

def rep_hash(t):
    return t.replace('#','xxhash ')

def rem_stop_words(t,):
    t = t.split()
    url_idx = [i for i,x in enumerate(t) if x in arb_stopwords] 
    
    for i in url_idx: t[i] = ''
    return ' '.join(t) 

def rem_punc(t):
    return t.translate(str.maketrans(' ', ' ', ''.join(elements.PUNCTUATION_MARKS)))

def rep_emojis(t):
    d = demoji.findall(t)
    if d:
        for k,v in d.items():
           t=t.replace(k,' xxemoji ') 
           #t=t.replace(k,' xxemoji xx'+v.replace(' ','_')+' ')
    return t

def rep_link(t):
    t_split = t.split()
    for i in range(len(t_split)):
        if 'http' in t_split[i]:
            t_split[i] = ' https '
    return ' '.join(t_split)

def rem_nonArabic_punc(t):
    l = list(elements.LETTERS)+[' ']
    l = ''.join([i for i in t if i not in l])
    l = ''.join([i for i in l if i.isnumeric() == False ])
    temp = []
    t = t.split()
    for i in range(len(t)):
        if t[i].startswith('xx'):
            temp.append(t[i])
        else :
            s = t[i].translate(str.maketrans(l,' '* len(l)))
            temp.append(s)
            
    return ' '.join(temp)

def normalization(t):
    translation = t.maketrans(elements.ALIF_MAQSURA, 'ي')
    t=t.translate(translation)

    translation = t.maketrans(elements.TA_MARBUTA,"ه")
    t=t.translate(translation)

    translation = t.maketrans(''.join(elements.NON_ALIF_HAMZA_FORMS),elements.HAMZA*2)
    t=t.translate(translation)

    translation = t.maketrans(''.join(elements.ALEF_HAMZA_FORMS),'ا'*3)
    return t.translate(translation)

def charachterize(t):
    full_text = ''
    for i in t.split():
        if i.startswith('xx') == True:
            full_text+= i
        else:
            full_text+=' '.join(list(i))
        full_text+=' $@'
    return full_text

def post_charachterize(t):
    return t.replace('$@',' ')
