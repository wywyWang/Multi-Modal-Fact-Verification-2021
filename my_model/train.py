from transformers import DebertaTokenizer, DebertaModel
import pandas as pd
import logging
import argparse
import pickle
import os
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
OUTPUT_PATH = './models/deberta_base_1/'
MAX_SEQUENCE_LENGTH = 512
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--output_folder_name",
                        type=str,
                        help="path to save model")
    opt.add_argument("--seed_value",
                        type=int,
                        default=42,
                        help="seed value")
    opt.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="batch size")
    opt.add_argument("--lr",
                        type=int,
                        default=5e-5,
                        help="learning rate")
    opt.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="epochs")
    config = vars(opt.parse_args())
    return config


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


class MultiModalDataset(Dataset):
    def __init__(self, nlp_tokenizer, nlp_pretrained, cv_pretrained, mode='train'):
        super().__init__()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225]),
        ])

        self.nlp_tokenizer = nlp_tokenizer
        self.nlp_pretrained = nlp_pretrained
        self.cv_pretrained = cv_pretrained

        with open('processed_{}.pickle'.format(mode), 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim_text, claim_image, document_text, document_image, category = self.data[idx+1]
        return (claim_text, claim_image, document_text, document_image, torch.tensor(category))


def save(model, config, epoch=None):
    output_folder_name = config['output_folder_name']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        model_name = output_folder_name + 'model'
        config_name = output_folder_name + 'config'
    else:
        model_name = output_folder_name + str(epoch) + 'model'
        config_name = output_folder_name + str(epoch) + 'config'
    
    torch.save(model.state_dict(), model_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))


if __name__ == '__main__':
    config = get_argument()
    set_seed(config['seed_value'])

    df_train = pd.read_csv('../data/train.csv', index_col='Id')[['claim', 'claim_image', 'document', 'document_image', 'Category']]
    df_val = pd.read_csv('../data/val.csv', index_col='Id')[['claim', 'claim_image', 'document', 'document_image', 'Category']]

    df_train['index'] = df_train.index
    df_val['index'] = df_val.index

    category = {
        'Support_Multimodal': 0,
        'Support_Text': 1,
        'Insufficient_Multimodal': 2,
        'Insufficient_Text': 3,
        'Refute': 4
    }

    df_train['Label'] = df_train['Category'].map(category)
    df_val['Label'] = df_val['Category'].map(category)

    # load pretrained NLP model
    deberta_tokenizer = DebertaTokenizer.from_pretrained(PRETRAINED_PATH)
    deberta = DebertaModel.from_pretrained(PRETRAINED_PATH)
    for param in deberta.parameters():
        param.requires_grad = False

    # load pretrained CV model
    vgg19_model = models.vgg19(pretrained=True)
    vgg19_model.classifier = vgg19_model.classifier[:-1]
    for param in vgg19_model.parameters():
        param.requires_grad = False

    fake_net = FakeNet()
    criterion = nn.CrossEntropyLoss()
    fake_net_optimizer = torch.optim.Adam(fake_net.parameters(), lr=5e-5)

    deberta.to(device)
    vgg19_model.to(device)
    fake_net.to(device)
    criterion.to(device)

    train_dataset = MultiModalDataset(nlp_tokenizer=deberta_tokenizer, nlp_pretrained=deberta, cv_pretrained=vgg19_model, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    # print(sum(p.numel() for p in deberta.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in vgg19_model.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in fake_net.parameters() if p.requires_grad))

    # training
    pbar = tqdm(range(config['epochs']), desc='Epoch: ')
    for epoch in pbar:
        fake_net.train()
        total_loss = 0
        for loader_idx, item in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # # transform sentences to embeddings via DeBERTa
            # input_claim = tokenizer(row['claim'], truncation=True, padding=True, return_tensors="pt").to(device)
            # output_claim = model(**input_claim)
            # claim_embedding = output_claim.last_hidden_state[:, 0, :]
            # claim_text_embedding = nlp_model(claim_embedding)

            # input_document = tokenizer(row['document'], truncation=True, padding=True, return_tensors="pt").to(device)
            # output_document = model(**input_document)
            # document_embedding = output_document.last_hidden_state[:, 0, :]
            # document_text_embedding = nlp_model(document_embedding)

            # # transform images to embeddings via VGG-19
            # path = '../data/train/'
            # filename = path + 'claim/' + row['Category'] + '/' + str(i) + '.jpg'
            # input_claim_image = Image.open(filename)
            # input_tensor = preprocess(input_claim_image).to(device)
            # input_batch = input_tensor.unsqueeze(0)
            # claim_image_embedding = cv_model(vgg19_model(input_batch))

            # filename = path + 'document/' + row['Category'] + '/' + str(i) + '.jpg'
            # input_document_image = Image.open(filename)
            # input_tensor = preprocess(input_document_image).to(device)
            # input_batch = input_tensor.unsqueeze(0)
            # document_image_embedding = cv_model(vgg19_model(input_batch))

            # # print(claim_image_embedding.shape, document_image_embedding.shape)
            # # print(claim_embedding.shape, document_embedding.shape)


            claim_text, claim_image, document_text, document_image, label = item[0], item[1].to(device), item[2], item[3].to(device), item[4].to(device)

            # transform sentences to embeddings via DeBERTa
            input_claim = deberta_tokenizer(claim_text, truncation=True, padding=True, return_tensors="pt").to(device)
            output_claim = deberta(**input_claim)
            output_claim_text = output_claim.last_hidden_state[:, 0, :]

            input_document = deberta_tokenizer(document_text, truncation=True, padding=True, return_tensors="pt").to(device)
            output_document = deberta(**input_document)
            output_document_text = output_document.last_hidden_state[:, 0, :]

            output_claim_image = vgg19_model(claim_image)

            output_document_image = vgg19_model(document_image)

            predicted_output = fake_net(output_claim_text, output_claim_image, output_document_text, output_document_image)
            
            loss = criterion(predicted_output, label)
            loss.backward()
            fake_net_optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_description("Loss: {}".format(round(current_loss, 3)), refresh=True)

    config['total_loss'] = total_loss
    save(fake_net, config)