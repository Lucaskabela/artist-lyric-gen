from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
# from transformers import AutoTokenizer, AutoModelWithLMHead

import os
import json
import re
import subprocess
import locale
from tqdm import tqdm
from lyrics import Lyrics

# pip install transformers
# pip install pronouncing

# Self -bleu measures diversity between verse
# Distinct-n measures diversity in a verse

loading_bar = None

def sel_bleu_artist(artist_corpus):
    '''
    Corpus is a list[list[str]], which is a list of verses.  Use to compute self-bleu
    '''
    total = 0
    list_of_references = []
    hypotheses = []
    for i in range(len(artist_corpus)):
        list_of_references.append(artist_corpus[:i] + artist_corpus[i+1:])
        hypotheses.append(artist_corpus[i])
    return corpus_bleu(list_of_references, hypotheses)

def sel_bleu_artist_avg(dataset):
    """
    dataset is a list[list[list[str]]], or a list of artist_corpus
    Returns per-artist self-bleu and dataset average
    """
    dataset_bleu = [sel_bleu_artist(artist) for artist in dataset]
    return dataset_bleu, sum(dataset_bleu) / len(dataset_bleu)


def distinct_n_verse(verse, n=1):
    """
    Calculates distinct n grams in a verse
    Verse should be list[str]
    """
    if len(verse) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(verse, n))
    return len(distinct_ngrams) / len(verse)

def distinct_n_artist(artist_corpus, n=1):
    """
    Corpus is list[list[str]], or a list of verses
    Returns average distinct-n for an artist
    """

    verses_d = [distinct_n_verse(verse, n) for verse in artist_corpus]
    return sum(verses_d) / len(artist_corpus)

def distinct_n(corpus, n=1):
    """
    Corpus is list[list[list[str]]], or a list of artist-verses
    Returns list of artist distinct-n and average
    """
    artist_d = [distinct_n_artist(artist, n) for artist in corpus]
    return artist_d, sum(artist_d) / len(corpus)

# def perplexity_artist(artist_corpus, tokenizer, model):
#     """
#     Returns the average perplexity of a artist-corpus using gpt2 as LM
#     """
#     ppls = []
#     for verse in artist_corpus:
#         tokenize_input = tokenizer.tokenize(verse)
#         tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#         loss=model(tensor_input, lm_labels=tensor_input)
#         ppls.append(math.exp(loss))
#     return sum(ppls) / len(ppls)


# TODO: Decide if this actually computes rhyme density,
def calc_rhyme_density(bars):
    """
    bars: list of bpe tokens
    """
    text = " ".join(bars).strip()
    text = bpe_string_to_text(text)
    # need to add quotes to the argument
    text = '"' + text + '"'
    params = ['java', '-jar', 'rhymeanalyzer/RhymeApp/dist/RhymeApp.jar', text]
    output = subprocess.check_output(params)

    # convert to string
    encoding = locale.getdefaultlocale()[1]
    output = output.decode(encoding)

    result = {}
    for line in output.split('\n'):
        dv = line.split(':')

        if len(dv) == 2:
            key = dv[0].strip()
            value = float(dv[1].strip())
            result[key] = value

    statistics = result
    return statistics['Rhyme_Density']

def dope_learning_rhyme_scores(bars):
    # TODO @bill: need to do real rhyme density, import from other repo
    text = " ".join(bars)
    text = bpe_string_to_text(text)
    l = Lyrics(text=text, language='en-us')
    rl = l.get_avg_rhyme_length()
    return rl

# def naive_rhyme_density(bars):
#     total_syllables = 0
#     rhymed_syllables = 0
#     words_used = set([word for bar in bars for word in bar.split()])
#     for bar in bars:
#         for word in bar.split():
#             p = pronouncing.phones_for_word(word)
#             if len(p) == 0:
#                 break
#             syllables = pronouncing.syllable_count(p[0])
#             total_syllables += syllables
#             has_rhyme = False
#             for rhyme in pronouncing.rhymes(word):
#                 if has_rhyme:
#                     break
#                 if rhyme in words_used:
#                     rhymed_syllables += syllables
#                     has_rhyme = True
#                     break
#     return rhymed_syllables/total_syllables


# TODO: Do artist similarity (cosine thing or crossentropy)

def rhyme_density(corpus):
    """
    Corpus is list[list[list[str]]], or a list of artist-verses
    Returns list of artist distinct-n and average
    """
    print("\n===== COMPUTING RD =====\n")
    rds = []
    global loading_bar
    loading_bar = tqdm(corpus)
    for artist in loading_bar:
        artist_rd = sum([calc_rhyme_density(verse) for verse in artist]) / len(artist)
        loading_bar.write(str(artist_rd))
        rds.append(artist_rd)
    return rds, sum(rds) / len(rds)

def get_lyric_blocks(song, input_format):
    if input_format == "raw_song":
        return [song['lyrics']]
    elif input_format == "verses":
        return [verse['lyrics'] for verse in song['verses']]
    return []

# def get_artist_to_verses_marked_verses():
#     songs_dir = os.path.join("./", "data", "songs", "marked_verses")
#     songs_file = os.path.join(songs_dir, "_LIST")
#     song_list = read_list_from_file(songs_file)
#     artist_to_verses = {}
#     for song_name in song_list:
#         song_file = name_to_file_name(song_name)
#         with open('{}/{}.json'.format(songs_dir, song_file)) as jsonfile:
#             song = json.load(jsonfile)
#             for verse in song['verses']:
#                 artists = verse['artists'][0]
#                 lyrics = verse['lyrics']
#                 if verse['valid']:
#                     if not artists in artist_to_verses:
#                         artist_to_verses[artists] = []
#                     artist_to_verses[artists].append(lyrics)
#     print("Done reading things up")
#     return artist_to_verses

def bpe_string_to_text(s):
    return re.sub(r'(@@ )|(@@ ?$)', '', s)

def clean_tokens(s):
    s = re.sub(r' L | L', '\n', s)
    return re.sub(r'S ', '', s)

def replace_start_with_new_line_from_list(l):
    return [token if token != 'S' else '\n' for token in l]

def remove_end_tokens_from_list(l):
    return list(filter(lambda x: x != 'L', l))

def get_artist_to_verses_model_output():
    filename = 'verses-2.json'
    with open(filename) as openfile:
        # this is in
        # {
            # 0 (artist id): [ (verses array)
            #     [(tokens array) a, b, c],
            # ]
        # }
        songs_json = json.load(openfile)
    artist_to_verses = {}
    for artist in songs_json:
        # Remove the S and replace L with \n
        artist_to_verses[artist] = [replace_start_with_new_line_from_list(remove_end_tokens_from_list(verse)) for verse in songs_json[artist]]
    return artist_to_verses

def get_artist_to_verses_test():
  return {1: [['a', 'b', 'c'], ['a' , 'z', 'b']], 2: ['hjkhkjhkjtdytr', 'asd']}

def write_out_stats_to_file(filename, stats_list):
    with open(filename) as openfile:
        json.dump(stats_list, openfile)

def main(file_prefix):
    artist_to_verses = get_artist_to_verses_model_output()
    per_artist_verses = artist_to_verses.values()
    artist = list(artist_to_verses.keys())
    rd, avg_rd = rhyme_density(per_artist_verses)
    write_out_stats_to_file('{}_rd.json'.format(file_prefix), rd)
    s_bleu, s_bleu_avg = sel_bleu_artist_avg(per_artist_verses)
    write_out_stats_to_file('{}_sbleu.json'.format(file_prefix), s_bleu)
    distinct_1s, distinct_1_avg = distinct_n(per_artist_verses, 1)
    write_out_stats_to_file('{}_d1.json'.format(file_prefix), distinct_1s)
    distinct_2s, distinct_2_avg = distinct_n(per_artist_verses, 2)
    write_out_stats_to_file('{}_d2.json'.format(file_prefix), distinct_2s)
    distinct_3s, distinct_3_avg = distinct_n(per_artist_verses, 3)
    write_out_stats_to_file('{}_d2.json'.format(file_prefix), distinct_3s)
    return (rd, s_bleu, distinct_1s, distinct_2s, distinct_3s)

if __name__ == "__main__":
    main('train_dataset')
