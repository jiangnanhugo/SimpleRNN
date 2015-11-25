import numpy
import random
import os
import stat
import subprocess
from os import chmod

PREFIX = os.getenv('ATISDATA', '')

def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename,'w')
    f.writelines(out)
    f.close()
    
    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = PREFIX + 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    
    # out = ['accuracy:', '16.26%;', 'precision:', '0.00%;', 'recall:', '0.00%;', 'FB1:', '0.00']
    
    precision = float(out[3][:-2])
    recall    = float(out[5][:-2])
    f1score   = float(out[7])

    return {'p':precision, 'r':recall, 'f1':f1score}


if __name__ == '__main__':
    print get_perf('valid.txt')
