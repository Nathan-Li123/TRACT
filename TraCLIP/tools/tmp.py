import os
import pickle
import numpy as np


if __name__ == '__main__':
    path = 'datasets/class_names/lvis_classes_with_attributes.txt'
    out_path = 'datasets/class_names/tmp.txt'
    texts  = np.loadtxt(path, delimiter='?', dtype=str).tolist()
    with open(out_path, 'w') as f:
        for text in texts:
            assert ':' in text
            name, description = text.split(':')
            out = f'\\noindent\n\\textbf{{{name}:}}\\textit{{{description}}}\n\n'
            f.write(out)
