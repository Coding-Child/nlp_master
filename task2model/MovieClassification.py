import torch.nn as nn
from model_BERT.bert import BERT

class MovieClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = BERT(self.config)
        # classfier
        self.projection_cls = nn.Linear(self.config.d_hidn, self.config.n_output, bias = False)

    def forward(self, inputs, segments):
        # (bs, n_enc_seq, d_hidn), (bs, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        # (bs, n_output)
        logits_cls = self.projection_cls(outputs_cls)
        # (bs, n_output), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return logits_cls, attn_probs