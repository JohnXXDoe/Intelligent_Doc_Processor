import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
from PIL import Image
import torch
torch.cuda.empty_cache()
#from spacy import displacy
import webbrowser
import tempfile

# For Flair
import flair
import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import logging
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from transformers import HfArgumentParser

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\33669\tesseract\tesseract.exe'
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
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'],
                             keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

# Flair Training data

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    layers: str = field(default="-1", metadata={"help": "Layers to be fine-tuned."})
    subtoken_pooling: str = field(
        default="first",
        metadata={"help": "Subtoken pooling strategy used for fine-tuned."},
    )
    hidden_size: int = field(
        default=256, metadata={"help": "Hidden size for NER model."}
    )
    use_crf: bool = field(
        default=False, metadata={"help": "Whether to use a CRF on-top or not."}
    )


@dataclass
class TrainingArguments:
    num_epochs: int = field(
        default=10, metadata={"help": "The number of training epochs."}
    )
    batch_size: int = field(
        default=8, metadata={"help": "Batch size used for training."}
    )
    mini_batch_chunk_size: int = field(
        default=1, metadata={"help": "If smaller than batch size, batches will be chunked."}
    )
    learning_rate: float = field(
        default=5e-05, metadata={"help": "Learning rate"}
    )
    seed: int = field(
        default=42, metadata={"help": "Seed used for reproducible fine-tuning results."}
    )
    device: str = field(default="cuda:0", metadata={"help": "CUDA device string."})
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for optimizer."}
    )
    embeddings_storage_mode: str = field(
        default="none", metadata={"help": "Defines embedding storage method."}
    )


@dataclass
class FlertArguments:
    context_size: int = field(
        default=0, metadata={"help": "Context size when using FLERT approach."}
    )
    respect_document_boundaries: bool = field(
        default=False,
        metadata={
            "help": "Whether to respect document boundaries or not when using FLERT."
        },
    )


@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "Flair NER dataset name."})
    dataset_arguments: str = field(
        default="", metadata={"help": "Dataset arguments for Flair NER dataset."}
    )
    output_dir: str = field(
        default="resources/taggers/ner",
        metadata={"help": "Defines output directory for final fine-tuned model."},
    )


def get_flair_corpus(data_args):
    ner_task_mapping = {}

    for name, obj in inspect.getmembers(flair.datasets.sequence_labeling):
        if inspect.isclass(obj):
            if (
                    name.startswith("NER")
                    or name.startswith("CONLL")
                    or name.startswith("WNUT")
            ):
                ner_task_mapping[name] = obj

    dataset_args = {}
    dataset_name = data_args.dataset_name

    if data_args.dataset_arguments:
        dataset_args = json.loads(data_args.dataset_arguments)

    if not dataset_name in ner_task_mapping:
        raise ValueError(
            f"Dataset name {dataset_name} is not a valid Flair datasets name!"
        )

    return ner_task_mapping[dataset_name](**dataset_args)


def flair_ner():
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = r'C:/Data'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt')

    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(model='roberta-base',
                                           layers="-1",
                                           subtoken_pooling="first_last",
                                           fine_tune=True,
                                           use_context=True,
                                           allow_long_sentences=True,
                                           )
    # embedding_types = [
    #     TransformerWordEmbeddings(model='roberta-base',
    #                               layers="-1",
    #                               subtoken_pooling="first_last",
    #                               fine_tune=True,
    #                               use_context=True,
    #                               allow_long_sentences=True
    #                               ),
    #     FlairEmbeddings('news-forward'),
    #     FlairEmbeddings('news-backward'),
    # ]

    #embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize bare-bones sequence tagger ()
    tagger = SequenceTagger(hidden_size=512,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            use_crf=True,
                            tag_type=label_type,
                            )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)


    # 7. run fine-tuning
    # trainer.fine_tune('resources/taggers/all-fixed-roberta-base',
    #                   learning_rate=5.0e-6,
    #                   mini_batch_size=2,
    #                   max_epochs=1000,
    #                   use_final_model_for_eval=False,
    #                   embeddings_storage_mode=None)
    # training
    # trainer.train('resources/taggers/reg_train',
    #               learning_rate=0.1,
    #               mini_batch_size=32,
    #               embeddings_storage_mode='none',
    #               max_epochs=200)
    path = 'resources/taggers/all-fixed-roberta-base'
    trained_model = SequenceTagger.load(path +'/best-model.pt')
    trainer.resume(trained_model,
                   base_path=path + '-resume',
                   max_epochs=1000,
                   )
if __name__ == '__main__':
    #   opt = get_config("config_files/en_filtered_config.yaml")
    # train(opt, amp=False)
    flair_ner()

