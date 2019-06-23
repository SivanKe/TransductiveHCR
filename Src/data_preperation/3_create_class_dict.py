import pickle
import os
import numpy as np
import argparse

def reduce_line_with_unknown_letters(transcription_file, unknown_letters):
    with open(transcription_file, 'r') as file:
        lines = file.readlines()
    splited_lines = [line.split("   *   ") for line in lines if len(line.split("   *   ")) > 1]
    bad_lines = [line for line in splited_lines if any(s in line[1] for s in unknown_letters)]
    good_lines = [line for line in splited_lines if not any(s in line[1] for s in unknown_letters)]
    with open(transcription_file.split('.')[0] + 'legal.txt', 'w') as file:
        good_lines = [str(rec[0]) + '   *   ' + str(rec[1]) for rec in good_lines]
        file.writelines(good_lines)
    with open(transcription_file.split('.')[0] + '_bad.txt', 'w') as file:
        bad_lines = [str(rec[0]) + '   *   ' + str(rec[1]) for rec in bad_lines]
        file.writelines(bad_lines)

def avg_unknown_letters(transcription_files, unknown_letters):
    text_lines = []
    for trans_file in transcription_files:
        with open(trans_file, 'r') as file:
            lines = file.readlines()
        text_lines = text_lines + [line.split("   *   ")[1] for line in lines if len(line.split("   *   ")) > 1]
    avg_unk_letters = sum([sum(line.count(s) for s in unknown_letters)/float(len(line)) for line in text_lines]) / float(len(text_lines))
    num_letters = np.array([float(len(line)) for line in text_lines])
    return avg_unk_letters,num_letters

def class_to_char(transcription_file):
    char_to_class = set()
    with open(transcription_file, 'r') as file:
        lines = file.readlines()
    splited_lines = [line.split("   *   ") for line in lines if len(line.split("   *   ")) > 1]
    #print(lines)
    #print([line.split("   *   ") for line in lines if len(line.split("   *   ")) <= 1])
    # print((str(pathlib.Path(base_dir) / splited_lines[0][0]), splited_lines[0][1]))

    ilegal_chars = ['&', '?', 'B', 'C', 'E', 'G', 'H', 'M', 'd', 'g', 'n', '{', '}', '\x81', '\x87', '«', '¹', '»', 'Ä', 'á']
    bad_lines = [line for line in splited_lines if any(s in line[1] for s in ilegal_chars)]
    good_lines = [line for line in splited_lines if not any(s in line[1] for s in ilegal_chars)]
    transcriptions = [line[1] for line in good_lines]
    with open(transcription_file, 'w') as file:
        good_lines = [str(rec[0]) + '   *   ' + str(rec[1]) for rec in good_lines]
        file.writelines(good_lines)
    with open(transcription_file + '_bad.txt', 'w') as file:
        bad_lines = [str(rec[0]) + '   *   ' + str(rec[1]) for rec in bad_lines]
        file.writelines(bad_lines)
    for trans in transcriptions:
        char_to_class |= set(trans.replace(" ", ""))
    char_to_class.remove('\n')
    return char_to_class

def create_class_dict(synth_transcriptions, orig_transcriptions, out_path, verbose=True):
    synth_dict = set()
    orig_dict = set()
    for synth in synth_transcriptions:
        synth_dict |= class_to_char(synth)
    for orig in orig_transcriptions:
        orig_dict |= class_to_char(orig)
    if verbose:
        non_trained_chars = orig_dict - synth_dict
        print(non_trained_chars)
        avg_unk_letters, num_letters = avg_unknown_letters(orig_transcriptions, non_trained_chars)
        print('len by percentile')
        print('range(10,100,10)')
        print([np.percentile(num_letters, per) for per in range(5, 100, 5)])
        print('mean unknown letter rate is: {}%'.format(avg_unk_letters * 100))
    synth_dict |= orig_dict

    if os.path.exists(out_path + '.pkl') or os.path.exists(out_path + '.txt'):
        raise Exception('Error: dictionary file already exists')

    with open(out_path + '.pkl', 'wb') as f:
        out_dict = dict([(l, i) for i, l in enumerate(sorted(synth_dict))])
        f.write(pickle.dumps(out_dict))

    with open(out_path + '.txt', 'w') as f:
        keys = ["'" + key + "'\n" for key in out_dict.keys()]
        f.writelines(keys)

    with open(out_path + '.pkl', 'rb') as f:
        print(pickle.loads(f.read()))

    return synth_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth_transcriptions', nargs='+',
                        default=[os.path.expandvars('../../Data/Synthetic/Prepared/data.txt')])
    parser.add_argument('--orig_transcriptions', nargs='+',
                        default=[os.path.expandvars('../../Data/Test/Prepared/im2line.txt')])
    parser.add_argument('--out_path', dest="out_path",
                        type=str, default=os.path.expandvars('../../Data/char_to_class'))
    args = parser.parse_args()

    out_dict = create_class_dict(args.synth_transcriptions, args.orig_transcriptions, args.out_path)



