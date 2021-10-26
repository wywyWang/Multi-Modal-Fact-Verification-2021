import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, PositionwiseFeedForward


class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 512

        self.text_embedding = nn.Sequential(
            nn.Linear(768, dim),
            nn.ReLU()
        )
        self.document_text_embedding = nn.Sequential(
            nn.Linear(768, dim),
            nn.ReLU()
        )

        self.image_embedding = nn.Sequential(
            nn.Linear(4096, dim),
            nn.ReLU()
        )
        self.document_image_embedding = nn.Sequential(
            nn.Linear(4096, dim),
            nn.ReLU()
        )

        self.self_attention = MultiHeadAttention(4, dim, dim, dim, dropout=0.1)
        self.pos_ffn = PositionwiseFeedForward(dim, dim*2, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(dim*4, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, claim_text, claim_image, document_text, document_image):
        claim_text_embedding = self.text_embedding(claim_text)
        claim_image_embedding = self.image_embedding(claim_image)
        document_text_embedding = self.document_text_embedding(document_text)
        document_image_embedding = self.document_image_embedding(document_image)

        claim_document_text, _ = self.self_attention(claim_text_embedding.unsqueeze(1), document_text_embedding.unsqueeze(1), document_text_embedding.unsqueeze(1))
        claim_document_text = self.pos_ffn(claim_document_text)
        claim_document_image, _ = self.self_attention(claim_image_embedding.unsqueeze(1), document_image_embedding.unsqueeze(1), document_image_embedding.unsqueeze(1))
        claim_document_image = self.pos_ffn(claim_document_image)

        claim_text_image, _ = self.self_attention(claim_text_embedding.unsqueeze(1), claim_image_embedding.unsqueeze(1), claim_image_embedding.unsqueeze(1))
        claim_text_image = self.pos_ffn(claim_text_image)
        document_text_image, _ = self.self_attention(document_text_embedding.unsqueeze(1), document_image_embedding.unsqueeze(1), document_image_embedding.unsqueeze(1))
        document_text_image = self.pos_ffn(document_text_image)
        
        concat_embeddings = torch.cat((claim_document_text.squeeze(1), claim_document_image.squeeze(1), claim_text_image.squeeze(1), document_text_image.squeeze(1)), dim=-1)
        predicted_output = self.classifier(concat_embeddings)
        return predicted_output