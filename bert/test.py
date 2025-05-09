from transformers import BertTokenizer
import os

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

#加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]
#
# #编码两个句子
# out = tokenizer.encode(
#     text=sents[0],
#     text_pair=sents[1],
#
#     #当句子长度大于max_length时,截断
#     truncation=True,
#
#     #一律补pad到max_length长度
#     padding='max_length',
#     add_special_tokens=True,
#     max_length=30,
#     return_tensors=None,
# )
#
# print(out)
#
# #增强的编码函数
# out = tokenizer.encode_plus(
#     text=sents[0],
#     text_pair=sents[1],
#
#     #当句子长度大于max_length时,截断
#     truncation=True,
#
#     #一律补零到max_length长度
#     padding='max_length',
#     max_length=30,
#     add_special_tokens=True,
#
#     #可取值tf,pt,np,默认为返回list
#     return_tensors=None,
#
#     #返回token_type_ids
#     return_token_type_ids=True,
#
#     #返回attention_mask
#     return_attention_mask=True,
#
#     #返回special_tokens_mask 特殊符号标识
#     return_special_tokens_mask=True,
#
#     #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
#     #return_offsets_mapping=True,
#
#     #返回length 标识长度
#     return_length=True,
# )
#
# #input_ids 就是编码后的词
# #token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
# #special_tokens_mask 特殊符号的位置是1,其他位置是0
# #attention_mask pad的位置是0,其他位置是1
# #length 返回句子长度
# for k, v in out.items():
#     print(k, ':', v)
#
#
# print(tokenizer.decode(out['input_ids']))
#
#
# #批量编码句子
# out = tokenizer.batch_encode_plus(
#     batch_text_or_text_pairs=[sents[0], sents[1]],
#     add_special_tokens=True,
#
#     #当句子长度大于max_length时,截断
#     truncation=True,
#
#     #一律补零到max_length长度
#     padding='max_length',
#     max_length=15,
#
#     #可取值tf,pt,np,默认为返回list
#     return_tensors=None,
#
#     #返回token_type_ids
#     return_token_type_ids=True,
#
#     #返回attention_mask
#     return_attention_mask=True,
#
#     #返回special_tokens_mask 特殊符号标识
#     return_special_tokens_mask=True,
#
#     #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
#     #return_offsets_mapping=True,
#
#     #返回length 标识长度
#     return_length=True,
# )
#
# #input_ids 就是编码后的词
# #token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
# #special_tokens_mask 特殊符号的位置是1,其他位置是0
# #attention_mask pad的位置是0,其他位置是1
# #length 返回句子长度
# for k, v in out.items():
#     print(k, ':', v)
#
# tokenizer.decode(out['input_ids'][0]), tokenizer.decode(out['input_ids'][1])
#

#批量编码成对的句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'][0])
