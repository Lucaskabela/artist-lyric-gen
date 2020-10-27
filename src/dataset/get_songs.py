import argparse
import lyricsgenius
import pandas as pd
import time
import sys
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file
from genius import GENIUS_ACCESS_TOKEN

raw_songs_dir = 'RAW_SONGS_DONT_DELETE'
artist_song_split_token = ' | '

artist_lyric_dir = 'raw_artist_lyrics'

def instantiate_genius():
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=30)
    genius.excluded_terms = ["Remix", "Live", "Intro", "Outro", "Freestyle", "Demo", "Interlude", "Snippet", "Excerpts", "Medley", "MTV", "Radio", "Edit", "Skit", "Discography"]
    return genius

def get_songs(name=None, csv=None):
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
    while len(artists) > 0:
        try:
            genius = instantiate_genius()
            # functions
            def process_artist(name, bar):
                artist = genius.search_artist(name)
                songs = artist.songs
                def process_song(song, bar):
                    return {
                        'title': song.title,
                        'artist': song.artist,
                        'lyrics': song.lyrics,
                        'featured_artists': [a['name'] for a in song.featured_artists]
                    }
                def get_song_name(song):
                    return song.artist + artist_song_split_token + song.title
                loop_and_process(songs, process_song, "Song", get_song_name, raw_songs_dir)
                return None
            def get_artist_name(name):
                return name
            loop_and_process(
                artists,
                process_artist,
                "Artist",
                get_artist_name,
                artist_lyric_dir,
            )
        except:
            e = sys.exc_info()[0]
            print(e)
        finally:
            completed_artists = read_list_from_file("{}/{}".format(artist_lyric_dir, "_LIST"))
            for artist in completed_artists:
                if artist in artists:
                    artists.remove(artist)

if __name__ == "__main__":
    get_songs(csv='get_artists.csv')

