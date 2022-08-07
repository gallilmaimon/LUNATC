import numpy as np

# this a small format change so that the vectors can be loaded as we want to
word2vec_dict = dict()
out_file = open("counter-fitted-vectors_formatted.txt", "wt")
out_file.write("65713 300" + "\n")  # size of vocabulary and vector size need to be added
with open("counter-fitted-vectors.txt", "r") as in_file:
    for i, line in enumerate(in_file):
        out_file.write(line)
        line_list = line.split()
        word2vec_dict[line_list[0]] = np.array(line_list[1:], dtype=np.float32)
out_file.close()
