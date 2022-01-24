import os
import datetime


def now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


repeat = 10

for r in range(repeat):
    for dataset in ['ml100k', 'wine', 'mnist']:
        commend = f"python ica_attack.py -d {dataset} -s {r} > log/{now()}_{dataset}_{r}.log"
        print(f'Running with {commend}')
        os.system(commend)
