from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModelWithLMHead
# pip install transformers
#pip install pronouncing


# For generation, can just use reference and 
def sel_bleu_gen(artist_list, generated):
    """
    artist list is list[str], name of artist
    generated is list[list[str]], list of generated lines
    """


def sel_bleu_artist(artist_corpus):
    '''
    Corpus is a list[list[str]], which is a list of lines.  Use to compute self-bleu
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
    """
    dataset_bleu = [sel_bleu_artist(artist) for artist in dataset]
    return sum(dataset_bleu) / len(dataset)



def distinct_n_sentence(sentence, n=1):
    """
    Should we define on the verse level instead?
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)

def distinct_n(corpus, n=1):
    sent_d = [distinct_n_sentence(sentence, n) for sentence in corpus]
    return sum(sent_d) / len(sentences)


def perplexity(corpus):
    """
    Returns the average perplexity of a corpus using gpt2 as LM
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    model.eval()
    ppls = []
    for sentence in corpus:
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=model(tensor_input, lm_labels=tensor_input)
        ppls.append(math.exp(loss))
    return sum(ppls) / len(ppls)


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
