import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
from PIL import Image
import pytesseract

import spacy
from spacy import displacy
import webbrowser
import tempfile

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\33669\tesseract\tesseract.exe'
cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

def train_ner():

    nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    doc = nlp(r"12.0 Cr was given to the company in compensation. In the period April 1 to September 20, 12.47 lakh cases of ticketless/irregular travellers were detected in suburban and non-suburban trains (long-distance trains) and an amount of `71.25 crore was realised as penalty. This is highest in terms of revenue among all zonal railways,” CR’s Chief Public Relations Officer Shivaji Sutar said.")
    print([(ent.text, ent.label_) for ent in doc.ents])
    visual = displacy.render(doc, style="ent")

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(visual)
    webbrowser.open(url)

if __name__ == '__main__':
    #opt = get_config("config_files/en_filtered_config.yaml")
    #train(opt, amp=False)
    train_ner()

