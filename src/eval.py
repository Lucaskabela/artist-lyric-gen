from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
# from transformers import AutoTokenizer, AutoModelWithLMHead
from dataset.dataset_utils import read_list_from_file, name_to_file_name
import os
import json
import pronouncing

# pip install transformers
# pip install pronouncing

# Self -bleu measures diversity between verse
# Distinct-n measures diversity in a verse

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
    dataset_bleu = [sel_bleu_artist(artist[:50]) for artist in dataset]
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

# def perplexity(corpus):
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     model = AutoModelWithLMHead.from_pretrained("gpt2")
#     model.eval()
#     artist_ppls = []
#     for artist in corpus:
#         artist_ppls.append(perplexity_artist(artist, tokenizer, model))
#     return artist_ppls, sum(artist_ppls)/len(artist_ppls)

# TODO: Decide if this actually computes rhyme density, 

# TODO: Do artist similarity (cosine thing or crossentropy)
def calc_rhyme_density(bars):
    """
    This seems sus...
    """
    total_syllables = 0
    rhymed_syllables = 0
    words_used = set([word for bar in bars for word in bar.split()])
    for bar in bars:
        for word in bar.split():
            p = pronouncing.phones_for_word(word)
            if len(p) == 0:
                break
            syllables = pronouncing.syllable_count(p[0])
            total_syllables += syllables
            has_rhyme = False
            for rhyme in pronouncing.rhymes(word):
                if has_rhyme:
                    break
                if rhyme in words_used:
                    rhymed_syllables += syllables
                    has_rhyme = True
                    break
    return rhymed_syllables/total_syllables 


def rhyme_density(corpus):
    """
    Corpus is list[list[list[str]]], or a list of artist-verses
    Returns list of artist distinct-n and average 
    """
    rds = []
    for artist in corpus:
        rds.append(sum([calc_rhyme_density(verse) for verse in artist[:50]]) / len(artist[:50]))
    return rds, sum(rds) / len(rds)

def get_lyric_blocks(song, input_format):
    if input_format == "raw_song":
        return [song['lyrics']]
    elif input_format == "verses":
        return [verse['lyrics'] for verse in song['verses']]
    return []

def main():
    songs_dir = os.path.join("./", "data", "songs", "marked_verses")
    songs_file = os.path.join(songs_dir, "_LIST")
    song_list = read_list_from_file(songs_file)
    artist_to_verses = {}
    for song_name in song_list:
        song_file = name_to_file_name(song_name)
        with open('{}/{}.json'.format(songs_dir, song_file)) as jsonfile:
            song = json.load(jsonfile)
            for verse in song['verses']:
                artists = verse['artists'][0]
                lyrics = verse['lyrics']
                if verse['valid']:
                    if not artists in artist_to_verses:
                        artist_to_verses[artists] = []
                    artist_to_verses[artists].append(lyrics)
    print("Done reading things up")
    per_artist_verses = artist_to_verses.values()
    artist = list(artist_to_verses.keys())
    rd, avg_rd = rhyme_density(per_artist_verses)
    artist_rd_max = artist[rd.index(max(rd))]
    artist_rd_min = artist[rd.index(min(rd))]
    s_bleu, s_bleu_avg = sel_bleu_artist_avg(per_artist_verses)
    artist_sb_max = artist[s_bleu.index(max(s_bleu))]
    artist_sb_min = artist[s_bleu.index(min(s_bleu))]
    distinct_1s, distinct_1_avg = distinct_n(per_artist_verses, 1)
    artist_d1_max = artist[distinct_1s.index(max(distinct_1s))]
    artist_d1_min = artist[distinct_1s.index(min(distinct_1s))]
    distinct_2s, distinct_2_avg = distinct_n(per_artist_verses, 2)
    artist_d2_max = artist[distinct_2s.index(max(distinct_2s))]
    artist_d2_min = artist[distinct_2s.index(min(distinct_2s))]
    distinct_3s, distinct_3_avg = distinct_n(per_artist_verses, 3)
    artist_d3_max = artist[distinct_3s.index(max(distinct_3s))]
    artist_d3_min = artist[distinct_3s.index(min(distinct_3s))]
    with open("out.txt", 'w') as out_file:
        out_file.write(str(artist_to_verses.keys()) + "\n")
        out_file.write(str(rd) + "\n")
        out_file.write("Average rhmye density is " + str(avg_rd)  + "\n")
        out_file.write("Maximum rhmye density is " + str(max(rd))  + " by " + artist_rd_max + "\n")
        out_file.write("Minimum rhmye density is " + str(min(rd))   + " by " + artist_rd_min + "\n")
        out_file.write(str(s_bleu) + "\n")
        out_file.write("Average self-bleu (4) is " + str(s_bleu_avg)  + "\n")
        out_file.write("Maximum self-bleu (4) is " + str(max(s_bleu))  + " by " + artist_sb_max + "\n")
        out_file.write("Minimum self-bleu (4) is " + str(min(s_bleu))  + " by " + artist_sb_min + "\n")
        out_file.write(str(distinct_1s) + "\n")
        out_file.write("Average distinct-1 is " + str(distinct_1_avg) + "\n")
        out_file.write("Maximum distinct-1 is " + str(max(distinct_1s))  + " by " + artist_d1_max + "\n")
        out_file.write("Minimum distinct-1 is " + str(min(distinct_1s))  +  " by " + artist_d1_min + "\n")
        out_file.write(str(distinct_2s) + "\n")
        out_file.write("Average distinct-2 is " + str(distinct_2_avg) + "\n")   
        out_file.write("Maximum distinct-2 is " + str(max(distinct_2s))   + " by " + artist_d2_max + "\n")
        out_file.write("Minimum distinct-2 is " + str(min(distinct_2s))  + " by " + artist_d2_min + "\n")     
        out_file.write(str(distinct_3s) + "\n")
        out_file.write("Average distinct-3 is " + str(distinct_3_avg) + "\n")
        out_file.write("Maximum distinct-3 is " + str(max(distinct_3s))  + " by " + artist_d3_max + "\n")
        out_file.write("Minimum distinct-3 is " + str(min(distinct_3s))  + " by " + artist_d3_min + "\n")

if __name__ == "__main__":
    main()
