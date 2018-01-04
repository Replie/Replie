# -*- coding utf-8 -*-
import logging
import random
import re

import time

import math
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from replie.models import Lang, EncoderRNN, AttnDecoderRNN

MAX_LENGTH = 20
SOS_TOKEN = 0
EOS_TOKEN = 1

log = logging.getLogger('reddit_q_a')
log.setLevel(logging.DEBUG)
hand = logging.FileHandler("log_reddit_q_a.log")
hand.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hand.setFormatter(formatter)
log.addHandler(hand)


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?,])", r" \1", s)
    s = re.sub(r"[^a-zA-Zא-ת.!?]+", r" ", s)
    return s


def read_question_answers(reverse=False):
    log.info("Reading lines...")

    # Read the file and split into lines
    question = open('questions.txt').read().strip().split('\n')
    answers = open('ans.txt').read().strip().split('\n')

    # Split every line into pairs and normalize
    # pairs = [[normalize_string(s) for s in question] normalize_string(s) for s in answers]

    pairs = zip([normalize_string(s) for s in question], [normalize_string(s) for s in answers])
    # pairs = zip(question, answers)
    # pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        lang = Lang('eng')
        # output_lang = Lang('eng')
    else:
        pairs = [list((p)) for p in pairs]
        lang = Lang('eng')
        # output_lang = Lang('eng')

    return lang, pairs


def print_pair(p):
    log.info(p[0])
    log.info(p[1])


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH #and p[1].startswith(good_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(reverse=False):
    lang, pairs = read_question_answers(reverse)
    log.info("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    log.info("Trimmed to %s sentence pairs" % len(pairs))

    log.info("Indexing words...")
    for pair in pairs:
        lang.index_sentence_by_word(pair[0])
        lang.index_sentence_by_word(pair[1])

    return lang, pairs


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return Variable(torch.LongTensor(indexes).view(-1, 1))


def variables_from_pair(pair):
    input_variable = variable_from_sentence(lang, pair[0])
    target_variable = variable_from_sentence(lang, pair[1])
    return (input_variable, target_variable)


def train(input_variable, target_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0].view(1, -1), target_variable[di])
            decoder_input = target_variable[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0].view(1, -1), target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_TOKEN: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(lang, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                     decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_randomly():
    pair = random.choice(pairs)

    output_words, decoder_attn = evaluate(pair[0])
    output_sentence = ' '.join(output_words)

    log.info(pair[0])
    log.info(pair[1])
    log.info(output_sentence)


teacher_forcing_ratio = 0.5
clip = 5.0


if __name__ == '__main__':
    lang, pairs = prepare_data()

    # Print an example pair
    print_pair(random.choice(pairs))

    # Testing

    # encoder_test = EncoderRNN(10, 10, 2)
    # decoder_test = AttnDecoderRNN('general', 10, 10, 2)
    #
    # encoder_hidden = encoder_test.init_hidden()
    #
    # word_input = Variable(torch.LongTensor([1, 2, 3]))
    # encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)
    #
    # word_inputs = Variable(torch.LongTensor([1, 2, 3]))
    # decoder_attns = torch.zeros(1, 3, 3)
    # decoder_hidden = encoder_hidden
    # decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))
    #
    # for i in range(3):
    #     decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context,
    #                                                                                  decoder_hidden, encoder_outputs)
    #     print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
    #     decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data
    #
    # teacher_forcing_ratio = 0.5
    # clip = 5.0

    log.info("Start Train")

    attn_model = 'general'
    hidden_size = 500
    n_layers = 2
    dropout_p = 0.05

    # Initialize models
    encoder = EncoderRNN(lang.n_words, hidden_size, n_layers)
    decoder = AttnDecoderRNN(attn_model, hidden_size, lang.n_words, n_layers, dropout_p=dropout_p)

    # Initialize optimizers and criterion
    learning_rate = 0.0001
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    n_epochs = 1000
    plot_every = 200
    print_every = 100

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Begin!
    for epoch in range(1, n_epochs + 1):

        # Get training data for this cycle
        training_pair = variables_from_pair(random.choice(pairs))
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # Run the train function
        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0:
            continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            percent = float(float(epoch) / float(n_epochs))
            print_summary = '%s (%d %d%%) %.4f' % (
                time_since(start, percent), epoch, epoch / n_epochs * 100, print_loss_avg)
            log.info(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    log.info("Done!")

    torch.save(encoder.state_dict(), "EncoderRNN.model")
    torch.save(decoder.state_dict(), "AttnDecoderRNN.model")

    for i in range(5):
        evaluate_randomly()
        log.info('\n')



