import argparse
import json
import datetime
import time
from tqdm import tqdm
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file
from get_songs import raw_songs_dir

# parser = argparse.ArgumentParser()
# parser.add_argument("--dir_path", type=str, help="Directory will all files")
# parser.add_argument("--list_file", type=str, help="List file with list of all songs")
# args = parser.parse_args()

char_allow_list = {'\n', ' '}

def get_context(lyrics, i):
    return lyrics[max(i - 10, 0): min(i + 10, len(lyrics))]

def get_lyric_blocks(song, input_format):
    if input_format == "raw_song":
        return [song['lyrics']]
    elif input_format == "verses":
        return [verse['lyrics'] for verse in song['verses']]
    return []

def analyze_characters(dir_path, list_file, input_format):
    song_list = read_list_from_file("{}/{}".format(dir_path, list_file))
    character_dict = {}
    j = 1
    start = time.time()
    bar = tqdm(song_list)
    for song_name in bar:
        # bar.write("starting {}, {} out of {}".format(song_name, j, len(song_list)))
        song_file_name = name_to_file_name(song_name.strip())
        with open('{}/{}.json'.format(dir_path, song_file_name)) as jsonfile:
            song = json.load(jsonfile)
            lyric_blocks = get_lyric_blocks(song, input_format)
            for lyrics in lyric_blocks:
                for i in range(0, len(lyrics)):
                    c = lyrics[i]
                    if not c.isalnum() and c not in char_allow_list:
                        if c not in character_dict.keys():
                            character_dict[c] = {
                                "count" : 1,
                                "context":
                                    [{"song": song_name, "line": get_context(lyrics, i)}]
                            }
                        else:
                            character_dict[c]['count'] = character_dict[c]['count'] + 1
                            character_dict[c]['context'].append({"song": song_name, "line": get_context(lyrics, i)})
        j = j + 1
    with open("character_stats.json", "w") as openfile:
        json.dump(character_dict, openfile)
    time_taken = str(datetime.timedelta(seconds=time.time() - start))
    print("{} for {}".format(time_taken, len(song_list)))
