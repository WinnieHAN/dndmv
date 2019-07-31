import subprocess
from multiprocessing import Pool
import os
import time
import numpy as np
import sys

train_langs = np.array(['eu', 'nl', 'fr', 'de', 'it', 'pl', 'pt', 'es', 'eu', 'nl', 'fr', 'de', 'it', 'pl', 'pt', 'es'])

#loaded_model_idx = sys.argv[1]
command1 = 'python src/eval_from_loaded_model.py --loaded_model_idx '
command2= '  --child_neural --em_type em --cvalency 2 --do_eval --ml_comb_type 0 ' +\
'--stc_model_type 1 --em_iter 1  --non_neural_iter 30 --non_dscrm_iter 60 --epochs 0 --function_mask ' #+\
#'--train et-la_ittb-no-fi-grc-nl-en-de-ja-bg-it-hi-fr-eu-sl --load_model --dev '

#print(command)
fstr = 'UAS is '
accs = []

for i in range(len(train_langs))[1:]:
    outputs = os.popen(command1 + str(i)+ command2 +' --train '+ train_langs[i] + ' --dev ' + train_langs[i])
    line = outputs.read()
    # print(line)
    if fstr in line:
        acc = line[line.index(fstr)+len(fstr):line.index(fstr)+len(fstr)+7]
        print(str(i)+": "+str(acc))
        accs.append(float(acc))

print('average: '+ str(sum(accs)/len(accs)))
