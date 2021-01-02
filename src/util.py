import torch
from torch.utils.data import Dataset
import os

class EnVietDataset(Dataset):
    def __init__(self, en_path, viet_path, en_vocab_path, viet_vocab_path):
        super().__init__()

        en_inputs = []
        viet_translations = []

        with open(en_path, 'r', encoding='utf-8') as en_f:
            for en_line in en_f.readlines():
                en_sequence = en_line.strip()
                en_tokens = en_sequence.split(' ')
                en_tokens.insert(0, '<s>')
                en_tokens.append('</s>')
                en_inputs.append(en_tokens)

        with open(viet_path, 'r', encoding='utf-8') as viet_f:
            for viet_line in viet_f.readlines():
                viet_sequence = viet_line.strip()
                viet_tokens = viet_sequence.split(' ')
                viet_tokens.insert(0, '<s>')
                viet_tokens.append('</s>')
                viet_translations.append(viet_tokens)

        # Vocab maps english tokens to indices then reverse vocab maps indices to english tokens
        en_vocab = self._build_vocab(en_vocab_path)
        en_reverse_vocab = {index: token for token, index in en_vocab.items()}

        # Vocab maps vietnamese tokens to indices then reverse vocab maps indices to vietnamese tokens
        viet_vocab = self._build_vocab(viet_vocab_path)
        viet_reverse_vocab = {index: token for token, index in viet_vocab.items()}

        self.en_vocab = en_vocab
        self.en_reverse_vocab = en_reverse_vocab

        self.viet_vocab = viet_vocab
        self.viet_reverse_vocab = viet_reverse_vocab

        indexed_en_inputs = [self.tokens_to_indices(en_input, lang='en') for en_input in en_inputs]
        indexed_viet_translations = [self.tokens_to_indices(viet_translation, lang='viet') for viet_translation in viet_translations]

        self.en_inputs = indexed_en_inputs
        self.viet_translations = indexed_viet_translations

    def __getitem__(self, index):
        return self.en_inputs[index], self.viet_translations[index]

    def __len__(self):
        return len(self.en_inputs)

    @staticmethod
    def _build_vocab(vocab_path):
        """Builds a vocab (dictionary) of word->index.

        Args:
            vocab_path (str): Path to the vocab.

        Returns:
            dict of word->index: The vocab of word->index.
        """
        assert os.path.exists(vocab_path)

        vocab = {'<pad>': 0}
        token_id = 1

        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                token = line.strip()
                vocab[token] = token_id
                token_id += 1

        return vocab

    def tokens_to_indices(self, tokens, lang='en'):
        """Converts a list of tokens from strings to their corresponding indices in the specified vocab.

        Args:
            tokens (list of str's): Tokens to be converted.
            lang (str, optional): Specifies which vocab to use. Defaults to 'en' for English. Other option
            is 'viet' for Vietnamese.

        Returns:
            length-N tensor: Tensor containing the indices corresponding to each token.
        """
        assert lang == 'en' or lang == 'viet'

        indices = []
        vocab = self.en_vocab if lang == 'en' else self.viet_vocab

        unk_token = vocab['<unk>']

        for token in tokens:
            indices.append(vocab.get(token, unk_token))

        return torch.tensor(indices)

    def indices_to_tokens(self, indices, lang='en'):
        """Converts indices to tokens and concatenates them as a string.

        Args:
            indices (list of str's): A tensor of indices (with shape (N, 1) or length-N), a list of (1, 1) tensors,
            or a list of indices (ints).
            lang (str, optional): Specifies which vocab to use. Defaults to 'en' for English. Other option
            is 'viet' for Vietnamese.

        Returns:
            str: String from concatenating the tokens.
        """
        assert lang == 'en' or lang == 'viet'

        tokens = []
        reverse_vocab = self.en_reverse_vocab if lang == 'en' else self.viet_reverse_vocab

        for index in indices:
            if torch.is_tensor(index):
                index = index.item()
            token = reverse_vocab.get(index, '<unk>')
            if token == '<pad>':
                continue
            tokens.append(token)

        return " ".join(tokens)

def collate_fn(batch):
    """Create a batch of data given a list of N input sequences and output sequences. Returns a tuple
    containing two tensors each with shape (N, max_sequence_length), where max_sequence_length is the
    maximum length of any sequence in the batch.

    Args:
        batch (list): A list of size N, where each element is a tuple containing two sequence tensors.

    Returns:
        tuple of two tensors, list of ints, list of ints: A tuple containing two tensors each with
    shape (N, max_sequence_length), list of each input sequence's length, and list of each target
    sequence's length.
    """
    en_inputs, viet_translations = zip(*batch)
    max_en_input_length = 0
    max_viet_translation_length = 0

    e = []
    v = []
    e_lens = []
    v_lens = []

    for en_input in en_inputs:
        en_input_length = list(en_input.size())[0]
        e_lens.append(en_input_length)
        if en_input_length > max_en_input_length:
            max_en_input_length = en_input_length
    for en_input in en_inputs:
        en_input_length = list(en_input.size())[0]
        if en_input_length < max_en_input_length:
            e.append(torch.cat((en_input, torch.zeros(max_en_input_length - en_input_length, dtype=int))))
        else:
            e.append(en_input)

    for viet_translation in viet_translations:
        viet_translation_length = list(viet_translation.size())[0]
        v_lens.append(viet_translation_length)
        if viet_translation_length > max_viet_translation_length:
            max_viet_translation_length = viet_translation_length
    for viet_translation in viet_translations:
        viet_translation_length = list(viet_translation.size())[0]
        if viet_translation_length < max_viet_translation_length:
            v.append(torch.cat((viet_translation, torch.zeros(max_viet_translation_length - viet_translation_length, dtype=int))))
        else:
            v.append(viet_translation)

    return (torch.stack(e), torch.stack(v)), e_lens, v_lens
