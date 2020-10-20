import json
from dataset_utils import loop_and_process, name_to_file_name
from get_songs import artist_lyric_dir
from tqdm import tqdm

raw_songs_dir = 'raw_songs'

def artist_to_raw_song_files(artists_file):
    with open(artists_file) as openfile:
        artists = openfile.readlines()
    artists = [artist.strip() for artist in artists]
    for artist_name in tqdm(artists):
        with open("{}/{}".format(artist_lyric_dir, name_to_file_name(artist_name))) as jsonfile:
            artist = json.load(jsonfile)
            songs = artist["songs"]
            def process_song(song):
                return {
                    'title': song['title'],
                    'artist': song['primary_artist']['name'],
                    'lyrics': song['lyrics']
                }
            def get_song_name(song):
                return song['title']
            loop_and_process(songs, process_song, "Song", get_song_name, raw_songs_dir)