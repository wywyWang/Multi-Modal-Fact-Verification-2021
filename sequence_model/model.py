import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, PositionwiseFeedForward


class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 512
        dropout = 0.1

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

        self.claim_document_text_attention = MultiHeadAttention(4, dim, dim, dim, dropout=dropout)
        self.claim_document_text_pos_ffn = PositionwiseFeedForward(dim, dim*2, dropout=dropout)
        self.claim_document_image_attention = MultiHeadAttention(4, dim, dim, dim, dropout=dropout)
        self.claim_document_image_pos_ffn = PositionwiseFeedForward(dim, dim*2, dropout=dropout)

        # self.text_image_attention = MultiHeadAttention(4, dim, dim, dim, dropout=dropout)
        # self.text_image_pos_ffn = PositionwiseFeedForward(dim, dim*2, dropout=dropout)
        # self.image_text_attention = MultiHeadAttention(4, dim, dim, dim, dropout=dropout)
        # self.image_text_pos_ffn = PositionwiseFeedForward(dim, dim*2, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(dim*6, dim),
            nn.ReLU(),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, claim_text, claim_image, document_text, document_image):
        # transform to embeddings
        claim_text_embedding = self.text_embedding(claim_text)
        claim_image_embedding = self.image_embedding(claim_image)
        document_text_embedding = self.document_text_embedding(document_text)
        document_image_embedding = self.document_image_embedding(document_image)

        # claim-document attention
        claim_document_text, _ = self.claim_document_text_attention(claim_text_embedding, document_text_embedding, document_text_embedding)
        claim_document_text = self.claim_document_text_pos_ffn(claim_document_text)
        claim_document_image, _ = self.claim_document_image_attention(claim_image_embedding.unsqueeze(1), document_image_embedding.unsqueeze(1), document_image_embedding.unsqueeze(1))
        claim_document_image = self.claim_document_image_pos_ffn(claim_document_image)

        # # text-image co-attention
        # claim_text_image, _ = self.text_image_attention(claim_text_embedding, claim_image_embedding, claim_image_embedding)
        # claim_text_image = self.text_image_pos_ffn(claim_text_image)
        # claim_image_text, _ = self.image_text_attention(claim_image_embedding, claim_text_embedding, claim_text_embedding)
        # claim_image_text = self.image_text_pos_ffn(claim_image_text)

        # document_text_image, _ = self.text_image_attention(document_text_embedding, document_image_embedding, document_image_embedding)
        # document_text_image = self.text_image_pos_ffn(document_text_image)
        # document_image_text, _ = self.image_text_attention(document_image_embedding, document_text_embedding, document_text_embedding)
        # document_image_text = self.image_text_pos_ffn(document_image_text)

        claim_document_text = torch.mean(claim_document_text, dim=1)
        claim_text_embedding = torch.mean(claim_text_embedding, dim=1)
        document_text_embedding = torch.mean(document_text_embedding, dim=1)
        
        concat_embeddings = torch.cat((claim_document_text, claim_document_image.squeeze(1),
                                       claim_text_embedding, claim_image_embedding,
                                       document_text_embedding, document_image_embedding), dim=-1)
        # concat_embeddings = torch.cat((claim_document_text, claim_document_image, 
        #                                claim_text_image, claim_image_text, 
        #                                document_text_image, document_image_text), dim=-1)
        predicted_output = self.classifier(concat_embeddings)
        return predicted_output