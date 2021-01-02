import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, input_tensor):
        """Forward pass through the encoder. The encoder processes one input sequence at a time.

        Note: In the case of this repository, we expect a batch of size 1 containing a length-seq_len tensor
        of the input sequence (e.g., the sequence to be translated).

        Args:
            input_tensor (1-by-seq_len tensor): Batch of size 1 containing a length-seq_len tensor
        of the input sequence (e.g., the sequence to be translated).

        Returns:
            tuple of two (1, N, hidden_size) tensors: (hn, cn) from the RNN (LSTM) layer.
        """
        N = list(input_tensor.size())[0]    # should be expecting batch size of 1 (1 sequence at a time)

        input_tensor = self.embedding(input_tensor)   # N-by-seq_len-by-embedding_dim
        out, (hn, cn) = self.rnn(input_tensor)    # out is N-by-seq_len-by-hidden_size, hn and cn are both 1-by-N-by-hidden_size

        return hn, cn

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, target_token, h0, c0):
        """Forward pass through the decoder. The decoder processes one target token at a time.

        Note: In the case of this repository, we expect a batch of size 1 containing the length-1 tensor of
        the target token in the target sequence (e.g., the translation).

        Args:
            target_token (1-by-1 tensor): Batch of size 1 containing the length-1 tensor of
        the target token in the target sequence (e.g., the translation).
            h0 (1-by-1-by-hidden_size tensor): Tensor containing the initial hidden state for each token in the batch.
            c0 (1-by-1-by-hidden_size tensor): Tensor containing the initial cell state for each token in the batch.

        Returns:
            tuple of one (1, vocab_size) tensor and two (1, 1, hidden_size) tensors: The predicted outputs
            and (hn, cn) from the RNN (LSTM) layer.
        """
        target_token = self.embedding(target_token)   # 1-by-1-by-embedding_dim
        out, (hn, cn) = self.rnn(target_token, (h0, c0))    # out is 1-by-1-by-hidden_size, hn and cn are both 1-by-1-by-hidden_size
        out = self.softmax(self.fc(out[0]))    # 1-by-vocab_size

        return out, hn, cn
