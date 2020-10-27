import json
import datetime
import time
import re
import traceback
import unidecode
from tqdm import tqdm

def name_to_file_name(name):
    return re.sub(r'\W+', '_', name)

def loop_and_process(
    objects, process_fn, item_type_name, get_obj_name_fn, data_dir, data_file_name_prefix=''
):
    i = 1
    start = time.time()
    os = tqdm(objects)
    for o in os:
        obj_name = get_obj_name_fn(o)
        obj_file_name = name_to_file_name(obj_name)
        try:
            # # os.set_description(
            #     "Starting {} {} out of {}: {}".format(
            #         item_type_name, i, len(objects), obj_name
            #     )
            # )
            processed_obj = process_fn(o, os)
            # os.set_description("Finished Processing {}".format(obj_name))
            # None means skip writing data to its own file, false means do not add to list
            if processed_obj is not None and processed_obj is not False:
                # write to own file
                with open(
                    "{}/{}{}.json".format(data_dir, data_file_name_prefix, obj_file_name), "w"
                ) as outfile:
                    json.dump(processed_obj, outfile)
                    # os.set_description("Wrote out data for {} to {}".format(obj_name, outfile.name))
            if processed_obj is not False:
                # add to the list
                with open("{}/_{}LIST".format(data_dir,data_file_name_prefix), "a") as outfile:
                    outfile.write("{}\n".format(obj_name))
                    # os.set_description("Wrote out {} to the list {}".format(obj_name, outfile.name))
            else:
                # removed from list
                with open("{}/_{}REMOVED".format(data_dir,data_file_name_prefix), "a") as outfile:
                    outfile.write("{}\n".format(obj_name))
                    # os.set_description("Wrote out {} to the removed list {}".format(obj_name, outfile.name))
            # os.set_description("Finished {}".format(obj_name))
        except Exception as e:
            tqdm.write("Failed to process {}".format(obj_name))
            with open("{}/_{}FAILED".format(data_dir,data_file_name_prefix), "a") as outfile:
                outfile.write("{}\n".format(obj_name))
            tqdm.write(e)
            traceback.print_exc()
        i = i + 1
    time_taken = str(datetime.timedelta(seconds=time.time() - start))
    print("{} for {} {}".format(time_taken, len(objects), item_type_name))

def read_list_from_file(list_path):
    with open(list_path) as listfile:
        l = listfile.readlines()
    return [i.strip() for i in l]

def write_list_to_file(l, list_path, mode):
    with open(list_path, mode) as outfile:
        for i in l:
            outfile.write("{}\n".format(i))

def remove_duplicates_from_list_file(dir_name, file_name):
    print('Removing duplicates from list {}/{}'.format(dir_name, file_name))
    l = read_list_from_file('{}/{}'.format(dir_name, file_name))
    s = list(set(l))
    s.sort()
    write_list_to_file(s, '{}/{}'.format(dir_name, file_name), 'w')
    print('Done removing duplicates')

def compress_spaces(s):
    s = re.sub(r' +', ' ', s)
    return s

def remove_special_characters(s):
    # letter changing
    s = unidecode.unidecode(s)
    s = s.lower()
    # dollar signs for numbers
    # removes commas from numbers
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # changes $100 to 100 dollar
    s = re.sub(r"(\$)([\d,\.]+)", r"\2 dollar ", s)
    # changes non dollar sign $ to s
    s = re.sub(r"\$", "s", s)
    # change % to percent
    s = re.sub("\%", " percent ", s)
    # @ to at
    s = re.sub(r"\@\'n", 'at\'n', s)
    s = re.sub(r"\@", ' at ', s)
    # #1 to number 1
    s = re.sub(r"(\#)(\d)", r" number \2", s)
    # these symbols get removed
    s = re.sub(r"\'|\′|\u2019|\u00b4|\u2032|\`|\u2018", "", s)
    # everything else to spaces
    s = re.sub(r"[^a-zA-Z0-9\n]", " ", s)
    s = compress_spaces(s)
    return s

def clean_artist_names(s):
    #TODO: get the way yo clean artist names
    s = s.lower()
    s = re.sub(r"6ix9ine", "six nine", s)
    s = re.sub(r"sixx-nine", "six nine", s)
    s = re.sub(r"ty\$", "ty dolla sign", s)
    s = re.sub(r"n9ne", "nine", s)
    s = remove_special_characters(s)
    return s