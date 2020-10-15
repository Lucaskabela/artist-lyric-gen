import argparse
import lyricsgenius
import pandas as pd
import time
from dataset_utils import loop_and_process, name_to_file_name
from genius import GENIUS_ACCESS_TOKEN

raw_songs_dir = 'raw_songs'
artist_song_split_token = ' | '

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
genius.excluded_terms = ["Remix", "Live", "Intro", "Outro", "Freestyle", "Demo", "Interlude", "Snippet"]
artist_lyric_dir = 'raw_artist_lyrics'

def get_songs(name=None, csv=None):
    def process_artist(name):
        artist = genius.search_artist(name, max_songs=3)
        songs = artist.songs
        def process_song(song):
            return {
                'title': song.title,
                'artist': song.artist,
                'lyrics': song.lyrics
            }
        def get_song_name(song):
            return song.artist + artist_song_split_token + song.title
        loop_and_process(songs, process_song, "Song", get_song_name, raw_songs_dir)
        return None

    def get_artist_name(name):
        return name

    artists = pd.DataFrame([], columns=['Artist'])
    if csv is not None:
        print("\n Getting lyrics for all artists in {}".format(csv))
        with open(csv) as openfile:
            artists = openfile.readlines()
        artists = [artist.strip() for artist in artists]
    elif name is not None:
        print("\n Getting lyrics for {}".format(name))
        artists = pd.DataFrame([name], columns=['Artist'])
    else:
        print("No Input Artists")
    loop_and_process(
        artists,
        process_artist,
        "Artist",
        get_artist_name,
        artist_lyric_dir,
    )

if __name__ == "__main__":
    get_songs(csv='get_artists.csv')

