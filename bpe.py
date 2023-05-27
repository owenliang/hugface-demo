from collections import defaultdict
from tokenizers import pre_tokenizers

corpus = [ # The first sentences from the abstract of "<Attention Is All You Need>"
    "The dominant sequence transduction models are based on complex recurrent orconvolutional neural networks that include an encoder and a decoder.",
    "The bestperforming models also connect the encoder and decoder through an attentionmechanism.",
    "We propose a new simple network architecture, the Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutionsentirely."
]
#################### Step1: word freq ################
word_freqs = defaultdict(int)
pre_tokenizer = pre_tokenizers.BertPreTokenizer()

for text in corpus:
    words_with_offsets = pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# defaultdict(<class 'int'>, {'The': 2, 'dominant': 1, 'sequence': 1, ...})

#################### Step2: alphabet ################
alphabet = [] # 字母表
for word in word_freqs.keys():
    if word[0] not in alphabet: # 是单词的第一个字母
        alphabet.append(word[0])
    for letter in word[1:]: # 不是单词的第一个字母
        if f"##{letter}" not in alphabet: # f"{letter}" 是格式化的语法，用 letter 变量的真实值来替代 {letter}
            alphabet.append(f"##{letter}")
alphabet.sort()

#print(alphabet)  
# ['##a', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##q', '##r', '##s', '##t', '##u', '##v', '##w', '##x', '##y', ',', '.', 'T', 'W', 'a', 'b', 'c', 'd', 'e', 'i', 'm', 'n', 'o', 'p', 'r', 's', 't', 'w']
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy() # add special token

#################### Step3: split word to char ################
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
} 
#print(splits) # 每个字符作为一个 subword
# {'The': ['T', '##h', '##e'], 'dominant': ['d', '##o', '##m', '##i', '##n', '##a', '##n', '##t'],...}  

#################### Step4: find highest score and merge ################

def compute_pair_scores(splits):
    ''' 计算每对相邻子词 merge 操作的得分
    
    :param splits: 截止到目前为止，每个单词的拆分
    '''
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1: # 只有一个子词（就是单词自身）
            letter_freqs[split[0]] += freq 
            continue
        for i in range(len(split) - 1): # 有多个子词
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq # 最后一个位置没有 pair，但是要处理
        
    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

def find_max_score(scores):
    ''' 计算得分最高的子词
    '''
    best_pair = ""
    max_score = None

    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    print("\t Find max score: pair[%s], freq[%s]"%(best_pair, max_score))
    return best_pair

def merge_pair(a, b, splits):
    ''' 子词合并，将当前 splits 中的所有 "a b" 形式的子词合并为 "ab"
    '''
    combine_ab = "%s%s"%(a,b[2:] if b.startswith("##") else b)
    
    for word in word_freqs:
        split = splits[word] # word 当前的子词拆分
        if len(split) == 1: # 子词只有一个，表示子词就是 word 自身
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b: # a 和 b 连续出现，可以合并
                split = split[:i] + [combine_ab, ] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

vocab_size = 50 

while len(vocab) < vocab_size:
    print("Current vocab size:%s"%len(vocab))
    scores = compute_pair_scores(splits)
    print("\t Top3 Pair scores:%s"% sorted(scores.items(),key=lambda x:-x[1])[:3]) # 得分降序排列
    current_pair = find_max_score(scores)
    new_subword = "%s%s"%(current_pair[0],current_pair[1][2:] if current_pair[1].startswith("##") else current_pair[1])
    splits = merge_pair(current_pair[0], current_pair[1], splits)
    print("\t Merge '%s %s' to '%s'"%(current_pair[0], current_pair[1], new_subword))
    vocab.append(new_subword)
# Current vocab size:46
#    Top3 Pair scores:[(('##q', '##u'), 0.1), (('##l', '##y'), 0.076923), (('t', '##h'), 0.072727)]
#    Find max score: pair[('##q', '##u')], freq[0.1]
#    Merge '##q ##u' to '##qu'    
# Current vocab size:47
#    Top3 Pair scores:[(('##l', '##y'), 0.076923), (('t', '##h'), 0.072727), (('b', '##a'), 0.066667)]
#    Find max score: pair[('##l', '##y')], freq[0.076923]
#    Merge '##l ##y' to '##ly'
# ...

#print(vocab) # 词表由 special token、初始字母表、以及 merge结果所组成
# ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '##a', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##q', '##r', '##s', '##t', '##u', '##v', '##w', '##x', '##y', ',', '.', 'T', 'W', 'a', 'b', 'c', 'd', 'e', 'i', 'm', 'n', 'o', 'p', 'r', 's', 't', 'w', '##qu', '##ly', 'th', 'Th']



def encode_word(word, vocab):
    ''' 用 WordPiece 对单词进行拆分
    '''
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab: # 最长匹配
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i]) # 匹配到的最长子词
        word = word[i:] # 拆剩下的
        if len(word) > 0:
            word = f"##{word}"
    return tokens

def tokenize(text, vocab):
    ''' 对文本进行 tokenize. vocab 为词表
    '''
    pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    pre_tokenize_result = pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    print(pre_tokenized_text)
    encoded_words = [encode_word(word, vocab) for word in pre_tokenized_text]
    return sum(encoded_words, []) # 对列表的列表进行 flatten 处理

print(tokenize("This's me  ." ,vocab))
# ['Th', '##i', '##s', '[UNK]', 's', 'm', '##e', '.']