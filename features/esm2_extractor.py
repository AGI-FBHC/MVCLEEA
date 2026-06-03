import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer


class ESM2Extractor:
    """
    ESM2 Feature Extractor

    Extracts 1280-dim mean-pooled embeddings from the last hidden layer
    of the pre-trained ESM2 model (esm2_t33_650M_UR50D).

    Args:
        model_name: HuggingFace model identifier
        device:     torch device
        max_length: Maximum tokenization length
    """

    def __init__(self, model_name: str = 'facebook/esm2_t33_650M_UR50D',
                 device: str = 'cpu', max_length: int = 1024):
        self.device = torch.device(device)
        self.max_length = max_length
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract(self, sequence: str) -> torch.Tensor:
        """
        Extract ESM2 embedding for a single protein sequence.

        Args:
            sequence: Amino acid sequence string
        Returns:
            embedding: (1280,) float32 tensor
        """
        inputs = self.tokenizer(
            sequence, return_tensors='pt',
            max_length=self.max_length, truncation=True, padding=False
        ).to(self.device)

        outputs = self.model(**inputs)
        # Mean pooling over sequence length (excluding special tokens)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        hidden = outputs.last_hidden_state
        embedding = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def extract_batch(self, sequences: list, batch_size: int = 8) -> torch.Tensor:
        """
        Extract ESM2 embeddings for a batch of sequences.

        Args:
            sequences:  List of amino acid sequence strings
            batch_size: Inference batch size
        Returns:
            embeddings: (N, 1280) float32 tensor
        """
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_seqs, return_tensors='pt',
                max_length=self.max_length, truncation=True, padding=True
            ).to(self.device)

            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            hidden = outputs.last_hidden_state
            embeddings = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
