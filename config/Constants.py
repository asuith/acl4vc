PAD = 0
UNK = 1
BOS = 2
EOS = 3
MASK = 4
VIS = 5

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
MASK_WORD = '<mask>'
VIS_WORD = '<vis>'

attribute_start = 6
attribute_end = 3006

base_checkpoint_path = './experiments'	# base path to save checkpoints
base_data_path = '../ACL4VC/NACF_VC_data/' # base path to load corpora and features

# opt_filename = 'opt_info.json' # the filename to save training/model options (settings)

# mapping of nltk pos tags
pos_tag_mapping = {}
content = [
    [["``", "''", ",", "-LRB-", "-RRB-", ".", ":", "HYPH", "NFP"], "PUNCT"],
    [["$", "SYM"], "SYM"],
    [["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"], "VERB"],
    [["WDT", "WP$", "PRP$", "DT", "PDT"], "DET"],
    [["NN", "NNP", "NNPS", "NNS"], "NOUN"],
    [["WP", "EX", "PRP"], "PRON"],
    [["JJ", "JJR", "JJS", "AFX"], "ADJ"],
    [["ADD", "FW", "GW", "LS", "NIL", "XX"], "X"],
    [["SP", "_SP"], "SPACE"], 
    [["RB", "RBR", "RBS","WRB"], "ADV"], 
    [["IN", "RP"], "ADP"], 
    [["CC"], "CCONJ"],
    [["CD"], "NUM"],
    [["POS", "TO"], "PART"],
    [["UH"], "INTJ"]
]
for item in content:
    ks, v = item
    for k in ks:
        pos_tag_mapping[k] = v
