import json
import datetime
import time
import re
import traceback
from tqdm import tqdm

def name_to_file_name(name):
    return re.sub(r'\W+', '_', name)

def loop_and_process(
    objects, process_fn, item_type_name, get_obj_name_fn, data_dir, data_file_name_prefix=''
):
    i = 1
    print("{} total {}".format(len(objects), item_type_name))
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
            processed_obj = process_fn(o)
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
                with open("{}/{}LIST".format(data_dir,data_file_name_prefix), "a") as outfile:
                    outfile.write("{}\n".format(obj_name))
                    # os.set_description("Wrote out {} to the list {}".format(obj_name, outfile.name))
            else:
                # removed from list
                with open("{}/{}REMOVED".format(data_dir,data_file_name_prefix), "a") as outfile:
                    outfile.write("{}\n".format(obj_name))
                    # os.set_description("Wrote out {} to the removed list {}".format(obj_name, outfile.name))
            # os.set_description("Finished {}".format(obj_name))
        except Exception as e:
            tqdm.write("Failed to process {}".format(obj_name))
            with open("{}/{}FAILED".format(data_dir,data_file_name_prefix), "a") as outfile:
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
    s = set(l)
    write_list_to_file(s, '{}/{}'.format(dir_name, file_name), 'w')
    print('Done removing duplicates')