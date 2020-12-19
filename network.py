import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 256

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.hidden_size = HIDDEN_SIZE

        # Construct embedding layer. Try embedding_dim = 64
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

    def forward(self, input_tensor):
        """[summary]

        Args:
            input_tensor ([1-by-seq_len tensor]): [description]

        Returns:
            [type]: [description]
        """
        N = list(input_tensor.size())[0]    # should be expecting batch size of 1 (1 sentence at a time)

        input_tensor = self.embedding(input_tensor)   # N-by-seq_len-by-embedding_dim
        out, (hn, cn) = self.rnn(input_tensor)    # out is N-by-seq_len-by-hidden_size, hn and cn are both 1-by-N-by-hidden_size

        return hn, cn

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.hidden_size = HIDDEN_SIZE

        # Construct embedding layer. Try embedding_dim = 64
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, target_token, h0, c0):
        """[summary]

        Args:
            target_token ([1-by-1 tensor]): [description]

        Returns:
            [type]: [description]
        """
        target_token = self.embedding(target_token)   # 1-by-1-by-embedding_dim
        out, (hn, cn) = self.rnn(target_token, (h0, c0))    # out is 1-by-1-by-hidden_size, hn and cn are both 1-by-1-by-hidden_size
        out = self.softmax(self.fc(out[0]))    # 1-by-vocab_size

        return out, hn, cn
