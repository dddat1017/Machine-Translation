import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import math

from util import EnVietDataset, collate_fn
from attention_network import AttnEncoderRNN, AttnDecoderRNN

def train(input_batch, target_batch, e_lens, v_lens, encoder, decoder, encoder_optim, decoder_optim, loss_fn, device):
    batch_loss_total = 0.0

    all_encoder_hidden_states, all_encoder_hn, all_encoder_cn = encoder(input_batch, e_lens)    # (N, seq_len, hidden_size), (1, N, hidden_size), (1, N, hidden_size)

    decoder_inputs = target_batch[:,0:1]    # N-by-1; the <s> from each sequence
    prev_hn = all_encoder_hn
    prev_cn = all_encoder_cn

    max_seq_len = max(v_lens)
    for time_step in range(max_seq_len - 1):
        outputs, hn, cn = decoder(decoder_inputs, prev_hn, prev_cn, all_encoder_hidden_states, device)

        loss = loss_fn(outputs, target_batch[:,time_step+1])
        batch_loss_total += loss.item()

        top_pred_vals, indices = outputs.topk(1)    # N-by-1 and N-by-1
        decoder_inputs = indices.detach()
        prev_hn = hn
        prev_cn = cn

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()

    return batch_loss_total

"""
def evaluate(input_seq, encoder, decoder, device):
    all_encoder_hidden_states, encoder_hn, encoder_cn = encoder(input_seq, [input_seq.shape[1]])
    encoder_hidden_states = torch.tensor(all_encoder_hidden_states[0].tolist())    # seq_len-by-hidden_size
    _, predicted_indices = decoder(encoder_hn, encoder_cn, encoder_hidden_states, device)

    return predicted_indices
"""

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return asMinutes(s)

if __name__ == "__main__":
    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print()

    en_path = './data/train.en.txt'
    viet_path = './data/train.vi.txt'
    en_vocab_path = './data/vocab.en.txt'
    viet_vocab_path = './data/vocab.vi.txt'
    batch_size = 100
    train_dataset = EnVietDataset(en_path, viet_path, en_vocab_path, viet_vocab_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    # Print out some random examples from the data
    print("Data examples:")
    random_indices = torch.randperm(len(train_dataset))[:8].tolist()
    for index in random_indices:
        en_indices, viet_indices = train_dataset.en_inputs[index], train_dataset.viet_translations[index]
        en_input = train_dataset.indices_to_tokens(en_indices, lang='en')
        viet_translation = train_dataset.indices_to_tokens(viet_indices, lang='viet')
        print(f"English: {en_input}. Vietnamese: {viet_translation}")
    print()

    learning_rate = 0.005
    momentum = 0.9
    embedding_dim = 256
    hidden_size = 512

    encoder = AttnEncoderRNN(len(train_dataset.en_vocab), embedding_dim, hidden_size)
    decoder = AttnDecoderRNN(len(train_dataset.viet_vocab), embedding_dim, hidden_size)

    encoder.to(device)
    decoder.to(device)

    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=momentum)
    decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=momentum)

    loss_fn = nn.NLLLoss()

    training_losses = []

    num_epochs = 1

    start = time.time()

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        for i, data in enumerate(train_loader):
            en, viet, e_lens, v_lens = data[0][0].to(device), data[0][1].to(device), data[1], data[2]
            total_batch_loss = train(en, viet, e_lens, v_lens, encoder, decoder, encoder_optim, decoder_optim, loss_fn, device)

            training_losses.append(total_batch_loss)
            print(f'[Epoch {epoch + 1}, Batch {i + 1} ({(i + 1) * batch_size} translations)] ({timeSince(start)}): {total_batch_loss}')

    plt.figure(1)
    plt.title('Loss per Batch')
    plt.xlabel(f'Batch (1 batch = {batch_size} translations)')
    plt.ylabel('Loss')
    plt.plot(training_losses)
    plt.show()

    torch.save(encoder.state_dict(), './attn_encoder.pth')
    torch.save(decoder.state_dict(), './attn_decoder.pth')

    """
    encoder.eval()
    decoder.eval()
    while True:
        en_input = input('> English: ')
        en_input_tokens = en_input.strip().split(' ')
        en_input_indices = train_dataset.tokens_to_indices(en_input_tokens, lang='en')
        test_en_input = en_input_indices.unsqueeze(0)
        test_en_input = test_en_input.long()
        with torch.no_grad():
            predicted_indices = evaluate(test_en_input, encoder, decoder, device)
            print(f'> Vietnamese: {train_dataset.indices_to_tokens(predicted_indices, lang="viet")}')
            print()
        keep_playing = input('> Translate something else? (y/n): ')
        if keep_playing == 'n': break
    """
