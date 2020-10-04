import argparse
import json
from dataset_utils import loop_and_process

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name of an artist")
parser.add_argument("--path", type=str, help="Path to the artist json file")
args = parser.parse_args()

raw_songs_dir = 'raw_songs'
raw_songs_file_prefix = 'Raw_Song'

if __name__ == "__main__":
    with open(args.path) as jsonfile:
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
        loop_and_process(songs, process_song, "Song", get_song_name, "{}/{}".format(raw_songs_dir, raw_songs_file_prefix))