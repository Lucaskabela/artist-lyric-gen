from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModelWithLMHead
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
    for i in range(len(corpus)):
        list_of_references.append(corpus[:i] + corpus[i+1:])
        hypotheses.append(corpus[i])
    return corpus_bleu(list_of_references, hyptohesises)

def sel_bleu_artist_avg(dataset):
    """
    dataset is a list[list[list[str]]], or a list of artist_corpus
    Returns per-artist self-bleu and dataset average
    """
    dataset_bleu = [sel_bleu_artist(artist) for artist in dataset]
    return dataset_bleu, sum(dataset_bleu) / len(dataset)



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

    verses_d = [distinct_n_sentence(verse, n) for verse in artist_corpus]
    return sum(verses_d) / len(artist_corpus)

def distinct_n(corpus, n=1):
    """
    Corpus is list[list[list[str]]], or a list of artist-verses
    Returns list of artist distinct-n and average 
    """
    artist_d = [distinct_n_artist(artist, n) for artist in corpus]
    return artist_d, sum(artist_d) / len(corpus)

def perplexity_artist(artist_corpus, tokenizer, model):
    """
    Returns the average perplexity of a artist-corpus using gpt2 as LM
    """
    ppls = []
    for verse in artist_corpus:
        tokenize_input = tokenizer.tokenize(verse)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=model(tensor_input, lm_labels=tensor_input)
        ppls.append(math.exp(loss))
    return sum(ppls) / len(ppls)

def perplexity(corpus):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    model.eval()
    artist_ppls = []
    for artist in corpus:
        artist_ppls.append(perplexity_artist(artist, tokenizer, model))
    return artist_ppls, sum(artist_ppls)/len(artist_ppls)

# TODO: Decide if this actually computes rhyme density, 

# TODO: Do artist similarity (cosine thing or crossentropy)
def calc_rhyme_density(bars):
    """
    This seems sus...
    """
    total_syllables = 0
    rhymed_syllables = 0
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
                for idx, r_bar in enumerate(bars):
                    if idx > 4:
                        break
                    if rhyme in r_bar:
                        rhymed_syllables += syllables
                        has_rhyme = True
                        break
   return rhymed_syllables/total_syllables 
