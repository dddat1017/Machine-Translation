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
    batch_loss = 0.0
    all_encoder_hidden_states, all_encoder_hn, all_encoder_cn = encoder(input_batch, e_lens)

    for idx, target_seq in enumerate(target_batch):
        encoder_hn = torch.tensor([[all_encoder_hn[0][idx].tolist()]])    # 1-by-1-by-hidden_size
        encoder_cn = torch.tensor([[all_encoder_cn[0][idx].tolist()]])    # 1-by-1-by-hidden_size
        encoder_hidden_states = torch.tensor(all_encoder_hidden_states[idx].tolist())    # seq_len-by-hidden_size
        preds, _ = decoder(encoder_hn, encoder_cn, encoder_hidden_states, device, cutoff=(v_lens[idx] - 1))

        loss = loss_fn(preds, torch.tensor(target_seq.tolist()[1 : preds.shape[0] + 1]))
        batch_loss += loss.item()

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()

    return batch_loss / input_batch.shape[0]

def evaluate(input_seq, encoder, decoder, device):
    all_encoder_hidden_states, encoder_hn, encoder_cn = encoder(input_seq, [input_seq.shape[1]])
    encoder_hidden_states = torch.tensor(all_encoder_hidden_states[0].tolist())    # seq_len-by-hidden_size
    _, predicted_indices = decoder(encoder_hn, encoder_cn, encoder_hidden_states, device)

    return predicted_indices

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
            en, viet, e_lens, v_lens = data[0][0], data[0][1], data[1], data[2]
            batch_loss = train(en, viet, e_lens, v_lens, encoder, decoder, encoder_optim, decoder_optim, loss_fn, device)

            training_losses.append(batch_loss)
            print(f'[Epoch {epoch + 1}, Batch {i + 1} ({timeSince(start)})]: {batch_loss}')
            if i == 2: break

    plt.figure(1)
    plt.title('Average Loss per Batch')
    plt.xlabel(f'Batch (1 batch = {batch_size} translations)')
    plt.ylabel('Average Loss')
    plt.plot(training_losses)
    plt.show()

    torch.save(encoder.state_dict(), './attn_encoder.pth')
    torch.save(decoder.state_dict(), './attn_decoder.pth')

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
