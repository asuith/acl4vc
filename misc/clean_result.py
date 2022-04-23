import sys
import pprint

import numpy as np

if __name__ == "__main__":
    # print("??")
    # pprint.pprint(sys.argv[1:])
    for input_file_path in sys.argv[1:]:
        output_file_path = input_file_path.replace(".txt", "_clean.txt")

        caption_length = []
        with open(input_file_path, "r") as fin, open(output_file_path, "w+") as fout:
            for line in fin.readlines():
                if line.startswith("video"):
                    words = line.split()
                    last = words[-1]
                    last = last[:last.index("(")]
                    words[-1] = last
                    fout.write(words[0] + "\n")
                    fout.write(" ".join(words[1:]) + "\n")
                    caption_length.append(len(words[1:]))
        np_lenth = np.array(caption_length)
        print("Average length for {}: {}".format(input_file_path, sum(caption_length) / len(caption_length)))

