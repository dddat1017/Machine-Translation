import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from util import EnVietDataset, collate_fn
from attention_network import AttnEncoderRNN, AttnDecoderRNN

def train(input_seq, target_seq, encoder, decoder, encoder_optim, decoder_optim, loss_fn, device):
    loss = 0.0
    encoder_hidden_states, decoder_h0, decoder_c0 = encoder(input_seq)

    target_tokens = target_seq[0][1:]    # all tokens except '<SOS>'
    decoder_input = target_seq[0][0]    # '<SOS>' token

    for token in target_tokens:
        output, hn, cn = decoder(torch.tensor([[decoder_input.item()]]).to(device), decoder_h0, decoder_c0, encoder_hidden_states, device)
        loss += loss_fn(output, torch.tensor([token.item()]).to(device))
        decoder_h0 = hn
        decoder_c0 = cn
        top_pred_val, top_pred_idx = output.topk(1)    # largest output and its corresponding index
        decoder_input = torch.tensor(top_pred_idx.item())
        if decoder_input.item() == 2: break    # predicted '<EOS>', so stop

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / target_tokens.shape[0]

def evaluate(input_seq, encoder, decoder, device):
    encoder_hidden_states, decoder_h0, decoder_c0 = encoder(input_seq)
    decoder_input = torch.tensor(1)

    predicted_indices = []

    # Model could potentially keep generating words on forever; so hacky stopping condition for now
    stopping_cond = 500

    for i in range(stopping_cond):
        output, hn, cn = decoder(torch.tensor([[decoder_input.item()]]).to(device), decoder_h0, decoder_c0, encoder_hidden_states)
        decoder_h0 = hn
        decoder_c0 = cn
        top_pred_val, top_pred_idx = output.topk(1)    # largest output and its corresponding index
        decoder_input = torch.tensor(top_pred_idx.item())
        predicted_indices.append(decoder_input.item())
        if decoder_input.item() == 2: break    # predicted '<EOS>', so stop

    return torch.tensor(predicted_indices)

if __name__ == "__main__":
    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print()

    en_path = './data/train.en.txt'
    viet_path = './data/train.vi.txt'
    en_vocab_path = './data/vocab.en.txt'
    viet_vocab_path = './data/vocab.vi.txt'
    train_dataset = EnVietDataset(en_path, viet_path, en_vocab_path, viet_vocab_path)
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

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
    running_loss = 0.0

    num_epochs = 1

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        for i, data in enumerate(train_loader):
            en, viet = data[0].to(device), data[1].to(device)
            loss = train(en, viet, encoder, decoder, encoder_optim, decoder_optim, loss_fn, device)
            running_loss += loss

            if i % 5000 == 4999:
                avg_loss = running_loss / 5000
                running_loss = 0.0
                training_losses.append(avg_loss)
                print(f'[Epoch {epoch + 1}, Batch {i + 1}]: {avg_loss}')

    plt.figure(1)
    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration (1 iteration = 1000 translations)')
    plt.ylabel('Loss')
    plt.plot(training_losses)
    plt.show()

    torch.save(encoder.state_dict(), './attn_encoder.pth')
    torch.save(decoder.state_dict(), './attn_decoder.pth')

    encoder.eval()
    decoder.eval()
    while True:
        en_input = input('> English: ')
        en_input_tokens = EnVietDataset._normalize(en_input.strip().split(' '))
        en_input_indices = train_dataset.tokens_to_indices(en_input_tokens, lang='en')
        test_en_input = torch.zeros((1, len(en_input_tokens)))
        test_en_input[0] = en_input_indices
        test_en_input = test_en_input.long()
        test_en_input = test_en_input.to(device)
        with torch.no_grad():
            predicted_indices = evaluate(test_en_input, encoder, decoder, device)
            print(f'> Vietnamese: {train_dataset.indices_to_tokens(predicted_indices, lang="viet")}')
            print()
        keep_playing = input('> Translate something else? (y/n): ')
        if keep_playing == 'n': break
