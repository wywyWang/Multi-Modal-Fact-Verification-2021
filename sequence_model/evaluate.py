from transformers import DebertaTokenizer, DebertaModel
# from transformers import RobertaTokenizer, RobertaModel
from transformers import ViTModel
import pandas as pd
import logging
import ast
import argparse
import pickle
import sys
import os
from sklearn.metrics import f1_score
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from model import FakeNet


transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

MODEL_TYPE = "deberta"
PRETRAINED_PATH = 'microsoft/deberta-base'
# PRETRAINED_PATH = 'roberta-base'
CV_PRETRAINED_PATH = 'facebook/deit-base-patch16-224'
OUTPUT_PATH = './models/deberta_base_1/'
MAX_SEQUENCE_LENGTH = 512
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


class MultiModalDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()

        # preprocess = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225]),
        # ])

        with open('../my_model/processed_{}.pickle'.format(mode), 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    #     claim_text, claim_image, document_text, document_image, category = self.data[idx+1]
    #     return (claim_text, claim_image, document_text, document_image, torch.tensor(category))

        claim_text, claim_image, document_text, document_image = self.data[idx+1]
        return (claim_text, claim_image, document_text, document_image)


if __name__ == '__main__':
    model_path = sys.argv[1]
    config = ast.literal_eval(open(model_path + '{}config'.format(sys.argv[2])).readline())
    set_seed(config['seed_value'])

    # df_val = pd.read_csv('../data/val.csv', index_col='Id')[['claim', 'claim_image', 'document', 'document_image', 'Category']]
    df_val = pd.read_csv('../data/test.csv', index_col='Id')[['claim', 'claim_image', 'document', 'document_image']]
    df_val['index'] = df_val.index

    category = {
        'Support_Multimodal': 0,
        'Support_Text': 1,
        'Insufficient_Multimodal': 2,
        'Insufficient_Text': 3,
        'Refute': 4
    }
    
    inverse_category = {
        0: 'Support_Multimodal',
        1: 'Support_Text',
        2: 'Insufficient_Multimodal',
        3: 'Insufficient_Text',
        4: 'Refute'
    }

    # df_val['Label'] = df_val['Category'].map(category)

    # load pretrained NLP model
    deberta_tokenizer = DebertaTokenizer.from_pretrained(PRETRAINED_PATH)
    deberta = DebertaModel.from_pretrained(PRETRAINED_PATH)
    # deberta_tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_PATH)
    # deberta = RobertaModel.from_pretrained(PRETRAINED_PATH)
    for param in deberta.parameters():
        param.requires_grad = False

    vit_model = ViTModel.from_pretrained(CV_PRETRAINED_PATH)
    for param in vit_model.parameters():
        param.requires_grad = False

    fake_net = FakeNet()

    fake_net.load_state_dict(torch.load(model_path + '{}model'.format(sys.argv[2])))

    deberta.to(device)
    vit_model.to(device)
    fake_net.to(device)

    # val_dataset = MultiModalDataset(mode='val')
    val_dataset = MultiModalDataset(mode='test')
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    # testing
    y_pred, y_true = [], []
    fake_net.eval()
    total_loss = 0
    for loader_idx, item in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # claim_text, claim_image, document_text, document_image, label = item[0], item[1].to(device), item[2], item[3].to(device), item[4].to(device)
        claim_text, claim_image, document_text, document_image = item[0], item[1].to(device), item[2], item[3].to(device)

        # transform sentences to embeddings via DeBERTa
        input_claim = deberta_tokenizer(claim_text, truncation=True, padding=True, return_tensors="pt").to(device)
        output_claim = deberta(**input_claim)
        output_claim_text = output_claim.last_hidden_state

        input_document = deberta_tokenizer(document_text, truncation=True, padding=True, return_tensors="pt").to(device)
        output_document = deberta(**input_document)
        output_document_text = output_document.last_hidden_state

        output_claim_image = vit_model(claim_image)
        output_claim_image = output_claim_image.last_hidden_state

        output_document_image = vit_model(document_image)
        output_document_image = output_document_image.last_hidden_state

        predicted_output = fake_net(output_claim_text, output_claim_image, output_document_text, output_document_image)
        softmax = nn.Softmax(dim=1)
        predicted_output = softmax(predicted_output)
        # _, predicted_label = torch.topk(predicted_output, 1)

        if len(y_pred) == 0:
            y_pred = predicted_output.cpu().detach().tolist()
            # y_true = label.tolist()
        else:
            y_pred += predicted_output.cpu().detach().tolist()
            # y_true += label.tolist()

    # f1 = round(f1_score(y_true, y_pred, average='weighted'), 5)
    
    # with open('record.csv', 'a') as config_file:
    #     config_file.write(model_path + ',' + str(f1))
    #     config_file.write('\n')

    answer = pd.DataFrame(y_pred, columns=category.keys())
    # answer['Category'] = answer['Category'].map(inverse_category)
    answer.to_csv('{}answer.csv'.format(model_path))