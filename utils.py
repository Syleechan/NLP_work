"""整个数据处理+模型训练过程中可能用到的工具函数"""
import re
import json


def recover_raw_text(text: str) -> str:
    """输入原始文本（含标注符号），返回从原始数据中解析得到的有效文本
    如果解析后的有效文本能与symptom标注中的所有位置对齐，认为该解析结果是有效的
        目前的解析方式会丢弃：
            * train.txt 中 1.9% 的特征；
            * dev.txt 中 9.7% 的特征
            * test.txt 中 5.4% 的特征
    """
    sp_text = list()

    # 抽取完整的症状分词结果
    nested, idx, span = 0, 0, ''
    while idx < len(text):
        if text[idx] == '[':
            if not nested and len(span) > 0:
                sp_text.append(span)
                span = ''
            nested += 1
            span += text[idx]
            idx += 1
            continue

        if text[idx] == ']':
            nested -= 1
            span += text[idx]
            idx += 1
            if not nested:
                sp_text.append(span + text[idx:idx+3])
                idx += 3
                span = ''
            continue

        span += text[idx]
        idx += 1

    assert not nested
    if len(span) > 0:
        sp_text.append(span)
    sp_text = [sp for sp in sp_text if len(sp) > 0]
    assert ''.join(sp_text) == text

    out_text = list()
    for span in sp_text:
        res = re.findall(r'\[\d+(.+?)\d+\](sym|dis|ite|bod)', span)
        if len(res) == 0:
            out_text.append(span.strip())
            continue

        # 处理嵌套的实体
        for r in res:
            tr = re.sub(r'\[(.+?)\](dis|ite|bod)', r'\1', r[0])
            out_text.append(tr)

    output = list()
    for span in out_text:
        if len(span) == 0:
            continue
        output.append(span.strip())
    return ''.join(output).replace(' ', '')


def convert_raw_to_data(in_path: str, out_path: str):
    """将最原始的数据格式转换并存储为易读的格式"""
    fin = open(in_path, 'r', encoding='utf-8')
    all_line = ''.join([r.strip() for r in fin.readlines()])
    all_line = json.loads(all_line)

    # 一条句子可能被存储成多条有效数据，因为每个句子中可能存在多种症状
    data = list()
    cnt, total_cnt = 0, 0
    mismatch, total_mismatch = 0, 0
    for line in all_line:
        text, symptom = line['text'], line['symptom']
        text = recover_raw_text(text)

        for sym, sym_value in symptom.items():
            cur_data = {'text': text, 'symptom': sym_value['self'], 'labels': list()}
            for key, value in sym_value.items():
                if isinstance(value, dict) and 'val' in value and 'pos' in value:
                    if len(value['val']) == 0 or len(value['pos']) == 0:
                        continue
                    total_cnt += 1
                    # 如果数据本身出现问题：特征数量与位置的长度对不上，则舍弃当前特征
                    try:
                        assert len(value['val'].split()) * 2 == len(value['pos'])
                    except Exception as err:
                        cnt += 1
                        continue
                    for idx, v in enumerate(value['val'].split()):
                        total_mismatch += 1
                        try:
                            start = value['pos'][idx * 2]
                            end = value['pos'][idx * 2 + 1] + 1
                            assert v == text[start:end]
                            cur_data['labels'].append((start, end - 1, key))
                        except:
                            mismatch += 1
                            if value['pos'][idx*2+1] - value['pos'][idx*2] + 1 != len(value['val']):
                                mismatch -= 1
                            continue
            data.append(cur_data)
    print('症状特征: {}个，无效{}个；共有特征属性数量：{}个，无法匹配数量{}个'.format(
        total_cnt, cnt, total_mismatch, mismatch
    ))

    # 将数据存储到指定地址
    fout = open(out_path, 'w', encoding='utf-8')
    length = list()
    for d in data:
        length.append(len(d['text']))
        fout.write(json.dumps(d, ensure_ascii=False) + '\n')
    fout.close()
    print('从原始数据条目：{}条中，抽取获得有效训练数据{}条。\n'.format(len(all_line), len(data)))
    length = list(sorted(length, key=lambda x: - x))
    return


if __name__ == '__main__':
    # 把原始数据转换为可以被bert读取的数据
    convert_raw_to_data('data/raw/dev.txt', 'data/dev.json')
    convert_raw_to_data('data/raw/test.txt', 'data/test.json')
    convert_raw_to_data('data/raw/train.txt', 'data/train.json')
