# Training Files

train.json, val.json, test.json are the files with our splits for training.

Each file is saved as a python list. Use the json package to load the list from
the file. Each object in the list represents a verse with the following
structure.

{
    artist_id: <artist id>
    lyrics: <string of the lyrics>
}

The lyrics is a full string of the BPE'd verses. We use 'S' (capital letter S),
as the start line token, and 'L' (capital letter L) as the end line token.

# BPE

bpe_string_token_to_int.json is a dict that can be loaded using the json
package. Given a string s, it will return an integer which represents which
token the string is.

Conversely, inn_to_bpe_string_token.json is a dict that given an integer,
returns a string of the token.

# Training

To do training, we can get the persona for the artist from either
persona_sentence_bpe.txt or persona_tags_bpe.txt.

We can split each verse into lines using the end of lines
tokens. Make sure to not remove the 'L' as we need to keep it there for
training.

A training example will be <persona, 'W', previous lines>, where 'W' (capital W)
is our special <start_song> token. Tokens must be separated by a single space.

Then we can split on ' ' (space) to separate out and get each token.
Then we use bpe_string_token_to_int.json to get the int for each token.