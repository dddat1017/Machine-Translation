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
        self.softmax = nn.LogSoftmax(dim=0)

    def forward_step(self, prev_output, prev_hn, prev_cn, encoder_hidden_states, device):
        """Performs a single step in the Decoder (i.e., compute the predictions, hn, and cn given a previous output, hn, and cn).

        Args:
            prev_output (1-by-1 tensor): The ouput from the previous time step.
            prev_hn (1-by-1-by-hidden_size tensor): The hidden state from the previous time step.
            prev_cn (1-by-1-by-hidden_size tensor): The cell state from the previous time step.
            encoder_hidden_states (seq_len-by-hidden_size tensor): The hidden states of the corresponding input sequence from the
        encoder, where seq_len is the length of the input sequence.

        Returns:
            tuple of one length-vocab_size tensor and two (1, 1, hidden_size) tensors: The predictions, hn, and cn from this time step.
        """
        prev_output = self.embedding(prev_output)    # 1-by-1-by-embedding_dim
        out, (hn, cn) = self.rnn(prev_output, (prev_hn, prev_cn))    # out is 1-by-1-by-hidden_size, hn and cn are both 1-by-1-by-hidden_size

        # Compute alignment scores then pass through softmax function
        # Take dot product between this decoder's ouputted hidden state and each hidden state of the input sequence from the encoder
        scores = torch.sum(encoder_hidden_states * out[0], dim=1)    # 1D tensor; length = input sequence's length
        scores = F.softmax(scores, dim=0)    # scores now lie in the range [0,1] and sum to 1

        # Weigh the input sequence's hidden states by the alignment scores then sum to get the context vector
        encoder_hidden_states_transposed = torch.transpose(encoder_hidden_states, 0, 1)
        context_vector = torch.matmul(encoder_hidden_states_transposed, scores)    # 1D tensor; length = hidden_size

        # Concatenate the decoder's ouputted hidden state with the context vector, then pass thru fc and softmax layers
        concat = torch.cat((out[0][0], context_vector), dim=0)    # 1D tensor; length = hidden_size*2
        concat = self.softmax(self.fc(concat))    # 1D tensor; length = vocab_size

        return concat, hn, cn

    def forward(self, encoder_hn, encoder_cn, encoder_hidden_states, device, cutoff=100):
        """Forward pass through the decoder.

        Args:
            encoder_hn (1-by-1-by-hidden_size tensor): Tensor containing the final hidden state of the corresponding input sequence.
            encoder_cn (1-by-1-by-hidden_size tensor): Tensor containing the final cell state of the corresponding input sequence.
            encoder_hidden_states (seq_len-by-hidden_size tensor): The hidden states of the corresponding input sequence from the
        encoder, where seq_len is the length of the input sequence.
            cutoff (int): Maximum number of predicted tokens expected. Defaults to 100.

        Returns:
            tuple of one (cutoff, vocab_size) tensor and one length-cutoff tensor: The predicted outputs
            and the indices that make up the translated sequence.
        """
        preds = []
        indices = []
        prev_output = torch.tensor([[2]])    # initial input, '<s>'
        prev_hn = encoder_hn
        prev_cn = encoder_cn
        for i in range(cutoff):
            pred, hn, cn = self.forward_step(prev_output, prev_hn, prev_cn, encoder_hidden_states, device)
            preds.append(pred)
            top_pred_val, top_pred_idx = pred.topk(1)    # most-likely prediction and its corresponding index
            indices.append(top_pred_idx.item())
            prev_output = torch.tensor([[top_pred_idx.item()]])
            prev_hn = hn
            prev_cn = cn
            if top_pred_idx.item() == 3: break    # predicted '</s>', so stop
        results = torch.stack(preds)
        return results, torch.tensor(indices)    # cutoff-by-vocab_size, length-cutoff
