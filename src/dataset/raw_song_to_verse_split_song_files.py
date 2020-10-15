import argparse
import json
from dataset_utils import loop_and_process, name_to_file_name
from artist_to_raw_song_files import raw_songs_dir

verse_split_songs_dir = "raw_verse_songs"

def raw_songs_to_verse_split_songs(song_list_path, dir_prefix):
    with open(song_list_path) as listfile:
        song_list = listfile.readlines()
    song_list = [song.strip() for song in song_list]
    def process_song(song):
        with open("{}/{}.json".format(raw_songs_dir, name_to_file_name(song))) as song_file:
            song = json.load(song_file)
        verses = []
        lyrics = song['lyrics']
        i = 0
        verse_lyrics = ''
        verse_metadata = ''
        while i < len(lyrics):
            # Parse the songs into each verse
            if lyrics[i] == '[':
                # we reached a new verse
                # append the previous verse if there was one
                if len(verse_metadata) > 0 and len(verse_lyrics) > 0:
                    verses.append({'metadata': verse_metadata, 'lyrics': verse_lyrics})
                # reset for the new verse
                verse_lyrics = ''
                verse_metadata = ''
                # start processing the new verse
                while i < len(lyrics) and lyrics[i] != ']':
                    verse_metadata = verse_metadata + lyrics[i]
                    i = i + 1
            else:
                verse_lyrics = verse_lyrics + lyrics[i]
            i = i + 1
        return {
            'title': song['title'],
            'verses': verses,
        }
    def get_song_name(song):
        return song
    loop_and_process(song_list, process_song, "Song", get_song_name, dir_prefix)

if __name__ == "__main__":
    raw_songs_to_verse_split_songs('raw_songs/LIST', verse_split_songs_dir)