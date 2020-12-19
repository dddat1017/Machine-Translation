import torch
from torch.utils.data import Dataset

import os
import pickle
import string
from collections import Counter

class EnVietDataset(Dataset):
    def __init__(self, path, en_vocab=None, en_reverse_vocab=None, viet_vocab=None, viet_reverse_vocab=None):
        super().__init__()

        en_inputs = []
        viet_translations = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                en, viet, _ = line.strip().split('\t')
                en_inputs.append(self._normalize(en.split(' ')))
                viet_translations.append(self._normalize(viet.split(' ')))

        # Vocab maps english tokens to indices
        if en_vocab is None:
            en_vocab = self._build_vocab(en_inputs, lang='en')
            en_reverse_vocab = None
        # Reverse vocab maps indices to english tokens
        if en_reverse_vocab is None:
            en_reverse_vocab = {index: token for token, index in en_vocab.items()}

        # Vocab maps vietnamese tokens to indices
        if viet_vocab is None:
            viet_vocab = self._build_vocab(viet_translations, lang='viet')
            viet_reverse_vocab = None
        # Reverse vocab maps indices to vietnamese tokens
        if viet_reverse_vocab is None:
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
    def _build_vocab(sentences, unk_cutoff=1, lang='en'):
        assert lang == 'en' or lang == 'viet'

        vocab_file_path = 'en_vocab.pkl' if lang == 'en' else 'viet_vocab.pkl'

        # Load cached vocab if existent
        if os.path.exists(vocab_file_path):
            with open(vocab_file_path, 'rb') as f:
                return pickle.load(f)

        word_counts = Counter()

        # Count unique words
        for sentence in sentences:
            sen = sentence[1 : len(sentence) - 1]    # don't count '<SOS>' and '<EOS>'
            for token in sen:
                word_counts[token] += 1

        # Special tokens: beginning of sentence and end of sentence
        vocab = {'[unk]': 0, '<SOS>': 1, '<EOS>': 2}
        token_id = 3

        # Assign a unique id to each word that occurs at least unk_cutoff number of times
        for token, count in word_counts.items():
            if count >= unk_cutoff:
                vocab[token] = token_id
                token_id += 1

        # Cache vocab
        with open(vocab_file_path, 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        return vocab

    @staticmethod
    def _normalize(sentence):
        result = [s.lower().translate(str.maketrans('', '', string.punctuation)) for s in sentence]
        result.insert(0, '<SOS>')
        result.append('<EOS>')
        return result

    def tokens_to_indices(self, tokens, lang='en'):
        assert lang == 'en' or lang == 'viet'

        indices = []
        vocab = self.en_vocab if lang == 'en' else self.viet_vocab

        unk_token = vocab['[unk]']

        for token in tokens:
            indices.append(vocab.get(token, unk_token))

        return torch.tensor(indices)

    def indices_to_tokens(self, indices, lang='en'):
        """
        Converts indices to tokens and concatenates them as a string.
        :param indices: A tensor of indices of shape (n, 1), a list of (1, 1) tensors or a list of indices (ints)
        :return: The string containing tokens, concatenated by a space.
        """
        assert lang == 'en' or lang == 'viet'

        tokens = []
        reverse_vocab = self.en_reverse_vocab if lang == 'en' else self.viet_reverse_vocab

        for index in indices:
            if torch.is_tensor(index):
                index = index.item()
            token = reverse_vocab.get(index, '[unk]')
            tokens.append(token)

        return " ".join(tokens)

def collate_fn(batch):
    en_inputs, viet_translations = zip(*batch)
    max_en_input_length = 0
    max_viet_translation_length = 0

    e = []
    v = []

    for en_input in en_inputs:
        en_input_length = list(en_input.size())[0]
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
        if viet_translation_length > max_viet_translation_length:
            max_viet_translation_length = viet_translation_length
    for viet_translation in viet_translations:
        viet_translation_length = list(viet_translation.size())[0]
        if viet_translation_length < max_viet_translation_length:
            v.append(torch.cat((viet_translation, torch.zeros(max_viet_translation_length - viet_translation_length, dtype=int))))
        else:
            v.append(viet_translation)

    return (torch.stack(e), torch.stack(v))
