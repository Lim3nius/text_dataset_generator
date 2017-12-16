from shutil import copyfile

import os
import sys


def get_last_file_index(directory):
    directory_path = os.path.realpath(directory)
    total_files = len([name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])
    last_index = (total_files - 1) / 2
    return last_index


def get_new_name(file_name, offset):
    file_name = file_name.split("/")[-1]
    name_splitted = file_name.split(".")
    name = name_splitted[0]
    ext = name_splitted[1]

    name_splitted = name.split("_");
    name = name_splitted[0]
    index = int(name_splitted[1])

    return name + "_" + str(index + offset) + "." + ext


def update_output(dir1, dir2, offset):
    content = []
    with open(dir1 + "/output.txt", "r") as f_read:
        for line in f_read:
            line_splitted = line.split("\t")
            name = line_splitted[0]
            value = line_splitted[1]
            content.append(get_new_name(name, offset) + "\t" + value)

    with open(dir2 + "/output.txt", "a") as f_append:
        for line in content:
            f_append.write(line + "\n")


def merge(dir1, dir2, offset):
    dir1 = os.path.realpath(dir1)
    dir2 = os.path.realpath(dir2)
    src_files = [os.path.join(dir1, name) for name in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, name)) and name != "output.txt"]
    
    for file_name in src_files:
        new_name = get_new_name(file_name, offset)
        copyfile(file_name, os.path.join(dir2, new_name))

    update_output(dir1, dir2, offset)


def main():
    src_directory = "Outputs_2"
    dst_directory = "Outputs"

    last_index = get_last_file_index(dst_directory + "/train/")
    merge(src_directory + "/train/", dst_directory + "/train/", last_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
