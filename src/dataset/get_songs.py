import argparse
import lyricsgenius
import pandas as pd
import time
from dataset_utils import loop_and_process
from genius import GENIUS_ACCESS_TOKEN

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name of an artist")
parser.add_argument("--csv", type=str, help="csv file with all artists to get")
args = parser.parse_args()

if __name__ == "__main__":
    def process_artist(row):
        _, row = row
        artist = genius.search_artist(row["Artist"])
        artist.save_lyrics()
        return {"artist": row["Artist"], "num_songs": artist.num_songs}

    def get_artist_name(row):
        _, row = row
        return row["Artist"]

    artists = pd.DataFrame([], columns=['Artist'])
    if args.csv is not None:
        print("\n Getting lyrics for all artists in {}".format(args.csv))
        artists = pd.read_csv(args.csv)
    elif args.name is not None:
        print("\n Getting lyrics for {}".format(args.name))
        artists = pd.DataFrame([args.name], columns=['Artist'])
    else:
        print("No Input Artists")
    loop_and_process(
        artists.iterrows(),
        process_artist,
        "Artist",
        get_artist_name,
        "Artists_Scraped",
    )

