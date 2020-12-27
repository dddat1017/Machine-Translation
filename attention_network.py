import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, batch_sequences, seq_lens):
        """Forward pass through the encoder.

        Args:
            batch_sequences (N-by-seq_len tensor): Batch containing N length-seq_len tensors
        (e.g., the sequences to be translated). N is the batch size.
            seq_lens (list of ints): List of sequences lengths of each batch element.

        Returns:
            tuple of one (N, seq_len, hidden_size) tensor and two (1, N, hidden_size) tensors: All hidden states of each
            sequence in the batch and (hn, cn) from the RNN (LSTM) layer.
        """
        batch_sequences = self.embedding(batch_sequences)    # N-by-seq_len-by-embedding_dim

        packed_batch_sequences = nn.utils.rnn.pack_padded_sequence(batch_sequences, lengths=seq_lens, batch_first=True, enforce_sorted=False)

        out, (hn, cn) = self.rnn(packed_batch_sequences)    # hn and cn are both 1-by-N-by-hidden_size

        # Unpack output from RNN (LSTM) layer. out_padded is N-by-seq_len-by-hidden_size
        out_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # out_padded: all hidden states of each sequence in the batch
        # hn: the final hidden state of each sequence in the batch
        # cn: final cell state of each sequence in the batch
        return out_padded, hn, cn

class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, prev_outputs, prev_hn, prev_cn, encoder_hidden_states, device):
        """Forward pass through the decoder.

        Args:
            prev_outputs (N-by-1): The ouputs from the previous time step. N is the batch size.
            prev_hn (1-by-N-by-hidden_size tensor):
            prev_cn (1-by-N-by-hidden_size tensor):
            encoder_hidden_states (N-input_seq_len-by-hidden_size tensor):

        Returns:
            tuple of one (N, vocab_size) tensor and two (1, N, hidden_size) tensors: The predicted outputs and (hn, cn) from
        the RNN (LSTM) layer.
        """
        embeddings = self.embedding(prev_outputs)    # N-by-1-by-embedding_dim
        out, (hn, cn) = self.rnn(embeddings, (prev_hn, prev_cn))    # out is N-by-1-by-hidden_size, hn and cn are both 1-by-N-by-hidden_size

        alignment_scores = torch.sum(encoder_hidden_states * out, dim=2)    # N-by-input_seq_len

        context_vectors = []
        for seq_hidden_state, score in zip(encoder_hidden_states, alignment_scores):
            sht = torch.tensor(torch.transpose(seq_hidden_state, 0, 1).tolist())    # hidden_size-by-input_seq_len
            c_vec = torch.matmul(sht.to(device), torch.tensor(score.tolist()).to(device))    # length-hidden_size
            context_vectors.append(c_vec)
        context_vectors = torch.stack(context_vectors)    # N-by-hidden_size

        concat = torch.cat((torch.tensor(torch.squeeze(out).tolist()).to(device), context_vectors), dim=1)    # N-by-hidden_size*2
        concat = self.softmax(self.fc(concat))    # N-by-vocab_size

        return concat, hn, cn
