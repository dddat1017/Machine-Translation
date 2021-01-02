# Machine-Translation

Solving the task of machine translation (from English to Vietnamese) with a regular Seq2Seq network and a global attention-based, dot product Seq2Seq network. Both following the Encoder-Decoder architecture.
#### Preprocessed data courtesy of the [Stanford NLP group](https://nlp.stanford.edu/projects/nmt/).

TODO:
- [] Attention-based network: Compute exact loss for each sequence (i.e., don't compute loss for sequences with padding). Look into utilizing `torch.nn.utils.rnn.pack_padded_sequence`
- [] Regular network: Get on track with the attention network, files are in `.src/TODO/`
- [] Training: Run for more epochs
- [] Training: Implement K-fold CV
- [] Training: Hyperparameter tuning
- [] Miscellaneous: More documentation + better organization

## Latest Results
---
### Attention-based network [01/01/2021]:
<img src="https://user-images.githubusercontent.com/40379856/103449606-44836d00-4c5f-11eb-87ee-d413873fb931.png" width="45%"></img>

## Standard Seq2Seq (Encoder-Decoder architecture) network
---
<img src="https://user-images.githubusercontent.com/40379856/102872974-56f6de80-43f5-11eb-9ea3-afb7ffc81162.jpg" width="90%"></img>

## Global attention-based, dot product Seq2Seq (Encoder-Decoder architecture) network
---
<img src="https://user-images.githubusercontent.com/40379856/102872988-5b22fc00-43f5-11eb-8eb5-c96b5cde2efc.gif" width="90%"></img>
