from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
# from transformers import AutoTokenizer, AutoModelWithLMHead

import os
import json
import re
import subprocess
import locale
from tqdm import tqdm, trange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np
from dataset import dataset_utils

# pip install transformers
# pip install pronouncing

# Self -bleu measures diversity between verse
# Distinct-n measures diversity in a verse

NUM_ARTISTS = 91
# ****REMEMBER: need to change this to the right location
RHYME_ANALYZER_JAR = 'rhymeanalyzer/RhymeApp/dist/RhymeApp.jar'

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

def sel_bleu_artist_avg(dataset, file_prefix):
    """
    dataset is a list[list[list[str]]], or a list of artist_corpus
    Returns per-artist self-bleu and dataset average
    """
    dataset_bleu = []
    for artist in dataset:
        artist_bleu = sel_bleu_artist(artist)
        append_stat_to_txt_file("{}_sbleu".format(file_prefix), artist_bleu)
        dataset_bleu.append(artist_bleu)
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

def distinct_n(corpus, n=1, file_prefix=''):
    """
    Corpus is list[list[list[str]]], or a list of artist-verses
    Returns list of artist distinct-n and average
    """
    artist_d = []
    for artist in corpus:
        artist_d_n = distinct_n_artist(artist, n)
        append_stat_to_txt_file("{}_d{}".format(file_prefix, n), artist_d_n)
        artist_d.append(artist_d_n)
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

def clean_lines(s):
    # Removes extra lines, and strips lines
    lines = [line.strip() for line in s.split('\n')]
    lines = list(filter(lambda s: s != '', lines))
    # reconstruct the lines together
    cleaned_lyrics = ''
    for line in lines:
        # concat lines together and add the end line token back
        cleaned_lyrics = cleaned_lyrics + line + '\n'
    return cleaned_lyrics

def calc_rhyme_density(bars):
    """
    bars: list of bpe tokens
    """
    text = " ".join(bars).strip()
    text = bpe_string_to_text(text)
    text = clean_lines(text)
    params = ['java', '-jar', RHYME_ANALYZER_JAR, text]
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
    return statistics['Rhyme_Density'] if 'Rhyme_Density' in statistics else 0

# def dope_learning_rhyme_scores(bars):
#     text = " ".join(bars)
#     text = bpe_string_to_text(text)
#     l = Lyrics(text=text, language='en-us')
#     rl = l.get_avg_rhyme_length()
#     return rl

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

def rhyme_density(corpus, file_prefix):
    """
    Corpus is list[list[list[str]]], or a list of artist-verses
    Returns list of artist distinct-n and average
    """
    rds = []
    for artist in tqdm(corpus):
        artist_rd = sum([calc_rhyme_density(verse) for verse in tqdm(artist)]) / len(artist)
        append_stat_to_txt_file("{}_rd".format(file_prefix), artist_rd)
        rds.append(artist_rd)
    return rds, sum(rds) / len(rds)

def calc_tfidf_score(gen_artist, vocab, dataset_artist):
    artist_dataset_verses_as_strings = [clean_lines(' '.join(verse)) for verse in dataset_artist]
    # combining all dataset lyrics to one string
    artist_dataset_all_verses = ' \n '.join(artist_dataset_verses_as_strings)

    all_verses_vectorizer = TfidfVectorizer(vocabulary = vocab)
    # vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    all_verses_dataset_vectors = all_verses_vectorizer.fit_transform([artist_dataset_all_verses])

    verse_vectorizer = TfidfVectorizer(vocabulary = vocab)
    verse_vectors = verse_vectorizer.fit_transform(artist_dataset_verses_as_strings)

    avgs = []
    maxs = []
    for i in trange(len(gen_artist), desc='Verse # for Artist', leave=False):
        verse = gen_artist[i]
        cleaned_verse = [clean_lines(' '.join(verse))]

        gen_to_all_verses_vector = all_verses_vectorizer.transform(cleaned_verse)
        all_verses_sim = cosine_similarity(gen_to_all_verses_vector, all_verses_dataset_vectors).flatten()

        gen_to_each_verse_vector = verse_vectorizer.transform(cleaned_verse)
        verse_max_sim = cosine_similarity(gen_to_each_verse_vector, verse_vectors).flatten()

        avgs.append(np.mean(all_verses_sim))
        maxs.append(np.max(verse_max_sim))
    return np.mean(avgs), np.mean(maxs)

def get_tfidf_scores(corpus, file_prefix):
    artist_to_verses_dataset = get_artist_to_verses_dataset('dataset/train.json')
    with open('dataset/bpe_string_token_to_int.json') as openfile:
        vocab = json.load(openfile).keys()
    average_tfidf_scores = []
    max_tfidf_scores = []
    for i in trange(len(corpus), desc='Artist #'):
        gen_artist = corpus[i]
        dataset_artist = artist_to_verses_dataset[i]
        avg_score, max_score = calc_tfidf_score(gen_artist, vocab, dataset_artist)
        append_stat_to_txt_file("{}_avg_sim_score".format(file_prefix), avg_score)
        append_stat_to_txt_file("{}_max_sim_score".format(file_prefix), max_score)
        average_tfidf_scores.append(avg_score)
        max_tfidf_scores.append(max_score)
    return average_tfidf_scores, max_tfidf_scores

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
    new_tokens = [token if token != 'S' else '\n' for token in l]
    if new_tokens[0] == '\n':
        new_tokens = new_tokens[1:]
    return new_tokens

def remove_end_tokens_from_list(l):
    return list(filter(lambda x: x != 'L', l))

def get_artist_to_verses_model_output(filename):
    with open(filename) as openfile:
        # this is in
        # {
            # 0 (artist id): [ (verses array)
            #     [(tokens array) a, b, c],
            # ]
        # }
        songs_json = json.load(openfile)
    artist_to_verses = []
    # instantiate list of lists, verses of artists
    for _ in range(0, NUM_ARTISTS):
        artist_to_verses.append([])
    for artist in songs_json:
        artist_index = int(artist) - 1
        # Remove the L and replace S with \n
        artist_to_verses[artist_index] = [replace_start_with_new_line_from_list(remove_end_tokens_from_list(verse)) for verse in songs_json[artist]]
    return artist_to_verses

def get_artist_to_verses_dataset(filename):
    with open(filename) as openfile:
        # {artist_id: ,
        # lyrics: <string>}
        verses_json = json.load(openfile)
    artist_to_verses = []
    # instantiate list of lists, verses of artists
    for _ in range(0, NUM_ARTISTS):
        artist_to_verses.append([])
    for verse in verses_json:
        artist_index = int(verse['artist_id']) - 1
        lyrics_tokens = verse['lyrics'].split(' ')
        cleaned_tokens = replace_start_with_new_line_from_list(remove_end_tokens_from_list(lyrics_tokens))
        artist_to_verses[artist_index].append(cleaned_tokens)
    return artist_to_verses

def get_artist_to_verses_test():
  return [[['a', 'b', 'c'], ['a' , 'z', 'b']], ['hjkhkjhkjtdytr', 'asd']]

def append_stat_to_txt_file(filename, stat):
    with open("{}.txt".format(filename), 'a') as openfile:
        openfile.write(str(stat))
        openfile.write('\n')

def write_out_stats_to_file(filename, stats_list):
    with open(filename, 'a') as openfile:
        json.dump(stats_list, openfile)

def main(file_prefix, read_file_name):
    """
    the files will be of the form <file_prefix>_<metric>.txt
    """
    # ****REMEMBER: you might need to change this to the correct function to
    # format rhyme density, most likely will be something like
    # get_artist_to_verses_model_output(<some file name>)
    # you may have to write your own function if the format is different from
    # the assumed format in that function
    artist_to_verses = get_artist_to_verses_dataset(read_file_name)
    per_artist_verses = artist_to_verses
    rd, avg_rd = rhyme_density(per_artist_verses, file_prefix)
    # ****REMEMBER: COMMENT THIS STUFF BELOW OUT IF YOU'RE ONLY DOING RHYME DENSITY
    s_bleu, s_bleu_avg = sel_bleu_artist_avg(per_artist_verses, file_prefix)
    distinct_1s, distinct_1_avg = distinct_n(per_artist_verses, 1, file_prefix)
    distinct_2s, distinct_2_avg = distinct_n(per_artist_verses, 2, file_prefix)
    distinct_3s, distinct_3_avg = distinct_n(per_artist_verses, 3, file_prefix)
    avg_sim_score, max_sim_score = get_tfidf_scores(per_artist_verses, file_prefix)
    return (rd, s_bleu, distinct_1s, distinct_2s, distinct_3s)

def list_to_floats(l):
    return [float(i) for i in l]

def summarize_metrics(file_prefix):
    # RD diff
    dataset_rds = list_to_floats(dataset_utils.read_list_from_file('train_dataset_rd.txt'))
    model_rds = list_to_floats(dataset_utils.read_list_from_file('{}_rd.txt'.format(file_prefix)))
    rd_diff = mean_squared_error(dataset_rds, model_rds)
    s_bleu = list_to_floats(dataset_utils.read_list_from_file('{}_sbleu.txt'.format(file_prefix)))
    d1 = list_to_floats(ataset_utils.read_list_from_file('{}_d1.txt'.format(file_prefix)))
    d2 = list_to_floats(dataset_utils.read_list_from_file('{}_d2.txt'.format(file_prefix)))
    d3 = list_to_floats(dataset_utils.read_list_from_file('{}_d3.txt'.format(file_prefix)))
    avg_sim = list_to_floats(dataset_utils.read_list_from_file('{}_avg_sim_score.txt'.format(file_prefix)))
    max_sim = list_to_floats(dataset_utils.read_list_from_file('{}_max_sim_score.txt'.format(file_prefix)))
    with open("{}_metrics_summary.txt".format(file_prefix), 'w') as openfile:
        openfile.write("RD Difference: {}\n".format(str(rd_diff)))
        openfile.write("Self BLEU: {}\n".format(str(np.mean(s_bleu))))
        openfile.write("Distinct 1: {}\n".format(str(np.mean(d1))))
        openfile.write("Distinct 2: {}\n".format(str(np.mean(d2))))
        openfile.write("Distinct 3: {}\n".format(str(np.mean(d3))))
        openfile.write("Avg Similarity: {}\n".format(str(np.mean(avg_sim))))
        openfile.write("Max Similarity: {}\n".format(str(np.mean(max_sim))))


if __name__ == "__main__":
    # ****REMEMBER: change the file name if necessary
    # main('train_dataset', 'dataset/train.json')
