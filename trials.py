import os
import datetime


def now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


repeat = 10

for r in range(repeat):
    for dataset in ['wine', 'mnist', 'ml100k']:
        os.system(f"python ica_attack.py -d {dataset} -s {r} > log/{now()}_{dataset}_{r}.log")
