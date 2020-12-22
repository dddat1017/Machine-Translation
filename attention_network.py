import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnEncoderRNN(nn.Module):
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
            tuple of one (1, seq_len, hidden_size) tensor and two (1, N, hidden_size) tensors: All the encoder's hidden
            states and (hn, cn) from the RNN (LSTM) layer.
        """
        N = list(input_tensor.size())[0]    # should be expecting batch size of 1 (1 sequence at a time)

        input_tensor = self.embedding(input_tensor)   # N-by-seq_len-by-embedding_dim
        out, (hn, cn) = self.rnn(input_tensor)    # out is N-by-seq_len-by-hidden_size, hn and cn are both 1-by-N-by-hidden_size

        return out, hn, cn

class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, target_token, h0, c0, encoder_hidden_states, device):
        """Forward pass through the decoder. The decoder processes one target token at a time.

        Note: In the case of this repository, we expect a batch of size 1 containing the length-1 tensor of
        the target token in the target sequence (e.g., the translation).

        Args:
            target_token (1-by-1 tensor): Batch of size 1 containing the length-1 tensor of
        the target token in the target sequence (e.g., the translation).
            h0 (1-by-1-by-hidden_size tensor): Tensor containing the initial hidden state for each token in the batch.
            c0 (1-by-1-by-hidden_size tensor): Tensor containing the initial cell state for each token in the batch.
            encoder_hidden_states (1-by-seq_len-by-hidden_size): Tensor containing all the hidden states from the
        encoder, where seq_len is the length of the input sequence.

        Returns:
            tuple of one (1, vocab_size) tensor and two (1, 1, hidden_size) tensors: The predicted outputs
            and (hn, cn) from the RNN (LSTM) layer.
        """
        target_token = self.embedding(target_token)    # 1-by-1-by-embedding_dim
        out, (hn, cn) = self.rnn(target_token, (h0, c0))    # out is 1-by-1-by-hidden_size, hn and cn are both 1-by-1-by-hidden_size

        input_seq_len = encoder_hidden_states[0].shape[0]    # input sequence's length

        # Compute alignment scores then pass through softmax function
        scores = torch.zeros((input_seq_len)).to(device)    # initialize with all zero's
        for i in range(input_seq_len):
            # Take dot product between each encoder's hidden states and this decoder's ouputted hidden state
            scores[i] = torch.dot(out[0][0], encoder_hidden_states[0][i])
        scores = F.softmax(scores, dim=0)    # scores now lie in the range [0,1] and sum to 1

        # Weigh the encoder's hidden states by the alignment scores then sum to get the context vector
        context_vector = torch.zeros((1, self.hidden_size)).to(device)    # 1-by-hidden_size
        for j in range(input_seq_len):
            context_vector[0] = context_vector[0].add(torch.mul(encoder_hidden_states[0][j], scores[j].item()))

        # Concatenate the decoder's ouputted hidden state with the context vector
        concat = torch.cat((out[0], context_vector), dim=1)    # 1-by-hidden_size*2

        # Pass through fc layer then softmax layer
        concat = self.softmax(self.fc(concat))    # 1-by-vocab_size

        return concat, hn, cn
