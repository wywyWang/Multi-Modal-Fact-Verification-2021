import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_embedding = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU()
        )

        self.image_embedding = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, claim_text, claim_image, document_text, document_image):
        claim_text_embedding = self.text_embedding(claim_text)
        claim_image_embedding = self.image_embedding(claim_image)
        document_text_embedding = self.text_embedding(document_text)
        document_image_embedding = self.image_embedding(document_image)
        
        concat_embeddings = torch.cat((claim_text_embedding, claim_image_embedding, document_text_embedding, document_image_embedding), dim=-1)
        predicted_output = self.classifier(concat_embeddings)
        return predicted_output