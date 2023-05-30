from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.models import WordPiece
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing

TRAIN=False 

sentence='大家好,欢迎来到小鱼儿teacher的直播间'

if TRAIN:
    # step1 normalizer: 中文按字，符号按字，英文按词，预处理一下字符串
    bertNormalizer=BertNormalizer()
    sentence=bertNormalizer.normalize_str(sentence)
    print('BertNormlizer:', sentence)

    # step2 pre_tokenizer: 按空格和标点符号分词
    bertPreTokenizer=BertPreTokenizer()
    tokens=bertPreTokenizer.pre_tokenize_str(sentence)
    print('bertPreTokenizer:', tokens)

    # step3 model：训练分词模型（对每个词拆字母，然后不断合并）
    tokenizer=Tokenizer(WordPiece(unk_token='[UNK]'))
    tokenizer.normalizer=bertNormalizer
    tokenizer.pre_tokenizer=bertPreTokenizer

    trainer=WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]) #,vocab_size=30000)
    tokenizer.train(['wikitext-103-raw/wiki.train.raw'],trainer)
    tokenizer.save('tokenizer-wiki.json')
else:
    # 基于wiki自训练的分词器
    tokenizer=Tokenizer.from_file('tokenizer-wiki.json')
    tokenizer.post_processor=TemplateProcessing(
        single='[CLS] $A [SEP]',
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ]
    )
    output=tokenizer.encode(sentence)
    print('wiki encode result...', 'tokens:', output.tokens, 'ids:', output.ids)

    # 演示反解
    print('decode:', tokenizer.decode(output.ids))
    print('token2id:', tokenizer.token_to_id('小'))
    print('id2token:', tokenizer.id_to_token(2018))

    # 对齐多个sentence长度,生成相应的注意力分数掩码
    tokenizer.enable_padding(pad_token='[PAD]')

    # 批量分词（为后续输入1个句子的NLP模型）
    output_list=tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"], )
    for output in output_list:
        print('pretrain batch encode result...', 'tokens:', output.tokens, 'ids:', output.ids, 'mask:',output.attention_mask)

    # 批量分词（为后续输入2个句子的NLP模型）
    output_list = tokenizer.encode_batch(
        [("Hello, y'all!", "How are you 😁 ?"), ("Hello to you too!", "I'm fine, thank you!")]
    )
    for output in output_list:
        print('pretrain batch encode pair result...', 'tokens:', output.tokens, 'ids:', output.ids)

    ############################################
    # 预训练的分词器(词更多更全)
    tokenizer=Tokenizer.from_pretrained('bert-base-chinese')
    output=tokenizer.encode(sentence)
    print('pretrain encode result...', 'tokens:', output.tokens, 'ids:', output.ids)