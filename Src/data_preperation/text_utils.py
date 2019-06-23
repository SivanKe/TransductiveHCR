import os, pathlib, subprocess
import numpy as np
import re

def remove_parentheses(test_str):
    ret = ''
    skip1c = 0
    skip2c = 0
    skip3c = 0
    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == '<':
            skip3c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif i == '>'and skip2c > 0:
            skip3c -= 1
        elif skip1c == 0 and skip2c == 0 and skip3c == 0 and i not in ')]>':
            ret += i

    return ret

def wylie2tibetan(wylie_text, tmp_dir,
                  perl_path='Lingua-BO-Wylie/bin'):
    tmp_dir = pathlib.Path(tmp_dir)
    page_before_path = str(tmp_dir / 'tmp_before.txt')
    page_after_path = str(tmp_dir / 'tmp_after.txt')

    with pathlib.Path(page_before_path).open('w') as pg_f:
        pg_f.writelines(wylie_text)
    args = ["perl", 'wylie.pl', page_before_path, page_after_path]
    out = subprocess.check_call(args, cwd=perl_path)
    if out != 0:
        print(out)
    os.remove(page_before_path)
    tibetan_text = pathlib.Path(page_after_path).read_text()
    os.remove(page_after_path)
    return tibetan_text

def wyliefile2tibetan(wylie_file, tibetan_file = None,
                  perl_path='Lingua-BO-Wylie/bin'):
    tmp_dir = pathlib.Path(wylie_file)
    if tibetan_file == None:
        tibetan_file = ('.').join(wylie_file.split('.')[:-1]) + '_tibetan.txt'

    args = ["perl", 'wylie.pl', wylie_file, tibetan_file]
    out = subprocess.check_call(args, cwd=perl_path)
    if out != 0:
        print(out)


def add_dot(text):
    if text[-1] != '་':
        return (text + '་')
    else:
        return text

def remove_parentheses(test_str):
    ret = ''
    skip1c = 0
    skip2c = 0
    skip3c = 0
    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == '<':
            skip3c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif i == '>'and skip2c > 0:
            skip3c -= 1
        elif skip1c == 0 and skip2c == 0 and skip3c == 0 and i not in ')]>':
            ret += i

    return ret

def create_wylie_rules_dict(rules_file):
    dict_rules = pathlib.Path(rules_file).read_text()
    line_start = len('convErr.Add("')
    dict_rules = [line for line in dict_rules.split("\n") if len(line) > 0]
    dict_rules = [line[line_start:-3] for line in dict_rules]
    dict_rules = dict([tuple(line.split('", "')) for line in dict_rules if len(line.split('", "')) > 1])
    keys = [re.sub("\\\\+", "", key) for key in dict_rules.keys()]
    values = [re.sub("\\\\+", "", value) for value in dict_rules.values()]
    dict_rules = dict(zip(keys, values))
    return dict_rules

def read_correct_wylie(file_path, rules_dict):
    pattern = re.compile('|'.join(rules_dict.keys()))
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()
        lines = [pattern.sub(lambda x: rules_dict[x.group()], line) for line in lines]
        lines = [remove_parentheses(line) for line in lines]
    return lines