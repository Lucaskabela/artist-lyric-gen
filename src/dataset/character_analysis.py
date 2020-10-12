import argparse
import json
import datetime
import time
from dataset_utils import loop_and_process, name_to_file_name
from artist_to_raw_song_files import raw_songs_dir, raw_songs_file_prefix

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, help="Directory will all files")
parser.add_argument("--list_file", type=str, help="List file with list of all songs")
args = parser.parse_args()

char_allow_list = {'\n'}

def get_context(lyrics, i):
    return lyrics[max(i - 10, 0): min(i + 10, len(lyrics))]

if __name__ == "__main__":
    song_list = []
    with open("{}/{}".format(args.dir_path, args.list_file)) as song_file:
        song_list = song_file.readlines()
    character_dict = {}
    j = 1
    start = time.time()
    for song_name in song_list:
        print("starting {}, {} out of {}".format(song_name, j, len(song_list)))
        song_file_name = name_to_file_name(song_name.strip())
        with open('{}/{}_{}.json'.format(raw_songs_dir, raw_songs_file_prefix, song_file_name)) as jsonfile:
            song = json.load(jsonfile)
            lyrics = song['lyrics']
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
