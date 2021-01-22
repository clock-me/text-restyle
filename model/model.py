import typing as tp

import torch
from torch import nn
from torch.nn import functional as functional


class LstmEncoder(nn.Module):
    """Simple bidirectional 2-layer LSTM encoder with temporal pooling"""
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 pool_window_size: int):
        """
        :hidden_size: size of hidden/cell states in LSTM
        :vocab_size: size of vocabulary. Used in embedding lookup tables
        :pool_window_size: size of window, used in temporal max pooling 
        """
        super().__init__()
        # storing params
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pool_window_size = pool_window_size
        # creating embedding lookup table
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=hidden_size)
        # creating 2-layer biLSTM
        self.rnn = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=2,
                           bidirectional=True)
        # creating temporal max-pooling
        # ceil mode for considering all the hidden states
        self.temporal_max_pool = nn.MaxPool1d(kernel_size=pool_window_size,
                                              ceil_mode=True)

    def forward(self,
                input_tensor: torch.Tensor,
                padding_mask: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs LSTM encoding & temporal max pooling
        :input_tensor: torch tensor of shape[input_length, batch_size]
        :padding_mask: False if place is padding, shape[input_length, batch_size]
        :return: 
            - encoded input, torch tensor of shape[ceil(input_length / pool_window_size), batch_size, 2 * hidden_size],
            - attn_mask, torch tensor shape[ceil(input_length / pool_window_size), batch_size,
        """
        # getting embeddings from input_tokens:
        input_embeddings = self.embedding(input_tensor)  # shape[input_length, batch_size, hidden_size]
        # encoding these embeddings with lstm:
        input_encoded, _ = self.rnn(input_embeddings)  # shape[input_length, batch_size, 2 * hidden_size]
        # before performing max pooling we should 
        # subtract some big number from padding positions to avoid considering padding tokens
        input_encoded = input_encoded - (~padding_mask[:, :, None]) * 2e9

        # some shape change
        input_encoded = input_encoded.permute(1, 2, 0)  # shape[batch_size, 2 * hidden_size, input_length]
        # performing max pooling
        input_maxpooled = self.temporal_max_pool(input_encoded)  # shape[batch_size, 2 * hidden_size, encoded_len]
        input_maxpooled = input_maxpooled.permute(2, 0, 1)  # shape[encoded_len, batch_size, 2 * hidden_size]

        # calculating old lengths for making attention mask
        lengths = torch.sum(padding_mask, dim=0, dtype=int) # shape[batch_size]
        # calculating new lengths for making attention mask
        lengths_after_pooling = torch.ceil(lengths / self.pool_window_size).type(torch.int64)

        # creating attention mask
        attn_mask = torch.arange(input_maxpooled.shape[0],
                                 device=lengths_after_pooling.device)[:, None] < lengths_after_pooling[None, :]

        return input_maxpooled, attn_mask


class Attention(nn.Module):
    """Attention mechanism, queries=values"""

    def __init__(self,
                 enc_size: int,
                 dec_size: int,
                 hid_size: int):
        """
        :enc_size: number of units in encoder state
        :dec_size: number of units in decoder state
        :hid_size: number of attention layer hidden units
        """
        super().__init__()
        # saving params:
        self.enc_size = enc_size  # number of units in encoder state
        self.dec_size = dec_size  # number of units in decoder state
        self.hid_size = hid_size  # number of attention layer hidden units

        self.linear_enc = nn.Linear(enc_size, hid_size, bias=False)  # for projecting encoder hidden states
        self.linear_dec = nn.Linear(dec_size, hid_size, bias=False)  # for projecting decoder hidden state
        self.linear_out = nn.Linear(hid_size, 1, bias=False)  # for producing attention logits

    def forward(self,
                enc_states: torch.Tensor,
                enc_mask: torch.Tensor,
                dec_state: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention response and weights.
        :enc_states: encoder activations, shape[n_enc, batch_size, enc_size]
        :enc_mask: padding mask, shape[n_enc, batch_size]
        :dec_state: single state of lstm decoder, shape[batch_size, dec_size]
        :return: 
            - attention response, shape[batch_size, enc_size]
            - attention weights, shape[n_enc, batch_size]
        """
        # projecting encoder activations:
        projected_enc_sequence = self.linear_enc(enc_states)  # shape[n_enc, batch_size, hid_size]
        # projecting decoder state
        projected_dec_state = self.linear_dec(dec_state)  # shape[batch_size, hid_size]
        # getting attention logits:
        activation = nn.functional.tanh(projected_enc_sequence + projected_dec_state[None, :, :])
        # ^-- shape [n_enc, batch_size, hid_size]
        attn_logits = self.linear_out(activation).squeeze(2)  # shape[n_enc, batch_size]
        # on padding positions logits should be small for not attending on them
        attn_logits = (attn_logits * enc_mask) - (~enc_mask) * 2e9
        attn_weights = nn.functional.softmax(attn_logits, dim=0)  # shape[n_enc, batch_size]
        attn_response = torch.sum(enc_states * attn_weights[:, :, None], dim=0)  # shape[batch_size, enc_size]
        return attn_response, attn_weights


class TransferModel(nn.Module):
    """
    Model for text style transfer.
    Described in https://arxiv.org/pdf/1811.00552v2.pdf
    The only difference is that this implementation supports only one
    style attribute;
    """

    def __init__(self,
                 hid_size: int,
                 pool_window_size: int,
                 vocab_size: int,
                 num_styles: int):
        """
        :hid_size: number of units in all hidden states of everything
            -- from embeddings to lstms/attention blocks.
        :pool_size: size of window in encoder's temporal pooling
        :vocab_size: number of words in language vocabulary
        :num_styles: number of different styles
        """
        super().__init__()
        self.hid_size = hid_size
        self.pool_window_size = pool_window_size

        self.encoder = LstmEncoder(hid_size, vocab_size, pool_window_size)
        self.attention = Attention(2 * hid_size, hid_size, hid_size)
        self.style_embeddings = nn.Embedding(num_styles, hid_size)
        self.decoder_cell = nn.LSTMCell(3 * hid_size, hid_size)
        self.out_linear = nn.Linear(hid_size, vocab_size)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self,
                corrupted_input: torch.Tensor,
                corrupted_input_mask: torch.Tensor,
                original_input: torch.Tensor,
                pad_mask: torch.Tensor,
                styles: torch.Tensor) -> torch.Tensor:
        """
        Computes reconstruction loss.
        Performs teacher-forcing approach, trying to restore corrupted input.
        :corrupted_input: torch tensor of shape[n_inp, batch_size], corrupted text,
            all numbers are in [0; vocab_size - 1]
        :corrupted_input_mask: torch tensor of shape[n_inp, batch_size], padding mask,
            of corrupted input, False if token is padding
        :original_input: torch tensor of shape[n_inp, batch_size], original text
            all numbers are in [0; vocab_size - 1]
        :pad_mask: torch tensor of shape[n_inp, batch_size], padding mask,
            zero if token is padding 
        :styles: torch tensor of shape[batch_size], styles
            all numbers are in [0; num_styles - 1]
        :return: Mean (by token) cross-entropy.
        """
        # encoding input with Encoder (described above)
        enc_crp_input, attn_mask = self.encoder(corrupted_input,
                                                corrupted_input_mask)
        # shape[n_enc, batch_size, 2 * hid_size], shape[n_enc, batch_size] 

        # getting start token embedding as embedding of styles 
        sos_token_embedding = self.style_embeddings(styles)  # shape[batch_size, hid_size]

        # also initial hidden states will be embeddings of styles

        dec_hid_state, dec_cell_state = sos_token_embedding, sos_token_embedding
        # ^-- shape[batch_size, hid_size], shape[batch_size, hid_size]

        # defining loss
        sum_loss = torch.tensor(0.0, device=self.style_embeddings.weight.device)
        num_non_pad_tokens = 0
        for t in range(len(original_input) - 1):
            # predicting current token:
            attention_response, _ = self.attention(enc_crp_input,
                                                   attn_mask,
                                                   dec_hid_state)  # shape[batch_size, 2 * hid_size]
            token_embedding_and_attn = torch.cat((
                attention_response,
                sos_token_embedding if t == 0 else self.encoder.embedding(original_input[t])
                # shape[batch_size, hid_size]
            ), dim=1)  # shape[batch_size, 3 * hid_size]
            dec_hid_state, dec_cell_state = self.decoder_cell(token_embedding_and_attn, (dec_hid_state, dec_cell_state))
            # predicting next token:
            next_token_distr = self.out_linear(dec_hid_state)  # shape[batch_size, vocab_size]
            # calculating loss
            loss = self.loss_func(next_token_distr, original_input[t + 1])  # shape[batch_size]
            sum_loss = sum_loss + (loss * pad_mask[t + 1]).sum()
            num_non_pad_tokens += int(pad_mask[t + 1].sum().item())

        sum_loss = sum_loss / num_non_pad_tokens
        return sum_loss

    def temperature_decode_batch_(self,
                                  enc_input: torch.Tensor,
                                  attn_mask: torch.Tensor,
                                  sos_embedding: torch.Tensor,
                                  temperature: float,
                                  max_steps: int,
                                  eos_token: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs sampling-with-temperature decoding
        :enc_input: Batch of encoded inputs.
            torch tensor of shape [n_inp, batch_size, 2 * hidden_size]
        :attn_mask: Attention mask.
            False means that decoder must not attend to position.
            torch tensor of shape [n_inp, batch_size]
        :sos_embedding:
            Start-of-sequence embeddings. Used for beginning of generation
            torch tensor of shape[batch_size, hid_size]
        :temperature: Temperature, applied to token classification logits
        :max_steps: Maximum steps for generation
        :eos_token: End of sequence token. Sequence in batch stops being generated if this token occurs.
        :returns:
            - torch tensor of shape [n_out, batch_size]
        """
        dec_hid_state, dec_cell_state = sos_embedding, sos_embedding  # shape[batch_size, hid_size]
        last_generated_token = sos_embedding  # shape[batch_size, hid_size]
        eos_generations = torch.zeros(last_generated_token.shape[0], dtype=torch.bool)

        generated_result = []
        for step in range(max_steps):
            attention_response, _ = self.attention(enc_input,
                                                   attn_mask,
                                                   dec_hid_state)  # shape [batch_size, hid_size]
            token_embedding_and_attn = torch.cat((
                attention_response,
                last_generated_token
            ), dim=1)  # shape[batch_size, 2 * hid_size]
            dec_hid_state, dec_cell_state = self.decoder_cell(token_embedding_and_attn,
                                                              (dec_hid_state, dec_cell_state))

            # ^-- shape[1, hid_size], shape[1, hid_size]
            next_token_distr = self.out_linear(dec_hid_state)  # shape[batch_size, vocab_size]
            # dividing next token_distribution by temperature
            next_token_distr = next_token_distr / temperature
            # softmax
            next_token_distr = nn.functional.softmax(next_token_distr, dim=1)
            generated_result.append(torch.multinomial(next_token_distr, 1).squeeze(1))
            eos_generations |= (generated_result == eos_token)
            if torch.all(eos_generations):
                break
        generated_result = torch.stack(generated_result, dim=0)
        eos_token_mask = torch.eq(generated_result, eos_token).type(torch.int64)
        cms_mask = torch.eq(torch.cumsum(eos_token_mask, dim=0), 0).type(torch.int64)
        lengths = cms_mask.sum(dim=0) + 1
        attn_mask = torch.less(torch.arange(generated_result.shape[0], device=generated_result.device)[:, None],
                               lengths[None, :])
        return generated_result, attn_mask

    def temperature_translate_batch(self,
                                    input_text: torch.Tensor,
                                    padding_mask: torch.Tensor,
                                    style: torch.Tensor,
                                    temperature: float,
                                    max_steps: int,
                                    eos_token: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs greedy decoding of batch of samples.
        :input_text: torch tensor of shape[n_inp, batch_size]
        :style: torch tensor of shape[batch_size]
        """
        enc_input, attn_mask = self.encoder(input_text, padding_mask)  # shape[n_inp, batch_size, 2 * hid_size]
        sos_embeddings = self.style_embeddings(style)  # shape[batch_size, hid_size]
        return self.temperature_decode_batch_(enc_input, attn_mask, sos_embeddings, temperature, max_steps, eos_token)
