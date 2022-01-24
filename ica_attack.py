import os
import argparse
import pdb
import numpy as np

from utils import generate_orthogonal_matrix
from ica import FastICA
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count
from functools import partial


def one_side_attack(px):
    fast_ica = FastICA(whiten=False, algorithm='deflation', n_components=px.shape[0])
    return fast_ica.fit_transform(px.T).T


def two_size_attack(pxq):
    xq = one_side_attack(pxq)
    return one_side_attack(xq.T).T


def correlation(real_x, rec_x, correlation_func=pearsonr, comp_func=np.max):
    m, n = real_x.shape
    results = []
    if m <= n:
        for i in range(real_x.shape[0]):
            with Pool(cpu_count()) as p:
                tmp_result = p.map(partial(correlation_func, real_x[i]), rec_x)
            results.append(comp_func([e[0] for e in tmp_result]))
        print(
            f'{getattr(correlation_func, "__name__")} '
            f'Mean {np.mean(results)} {getattr(comp_func, "__name__")} {comp_func(results)}'
        )
    else:
        return correlation(real_x.T, rec_x.T)
    return results


def root_mse(a, b):
    return [np.sqrt(np.mean(np.square(a - b)))]


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data', '-d', help='dataset', choices=('wine', 'mnist', 'ml100k'), default='ml100k')
    args_parser.add_argument('--seed', '-s', help='random seed', type=int, default=0)
    args = args_parser.parse_args()

    print(args)

    x = np.load(os.path.join('data', args.data + '.npy'))

    print('Data shape', x.shape)

    m, n = x.shape

    np.random.seed(args.seed)

    p = generate_orthogonal_matrix(m, block_reduce=1000)
    q = generate_orthogonal_matrix(n, block_reduce=1000)

    pxq = np.zeros(x.shape)
    index = 0
    for i in range(len(p)):
        pxq[index:index+len(p[i])] = p[i] @ x[index:index+len(p[i])]
        index += len(p[i])

    print('#' * 30)
    print('Attack PX')

    dec_x = one_side_attack(pxq)
    np.save(os.path.join('data', args.data + f'_px_{args.seed}' + '.npy'), dec_x)
    correlation(real_x=x, rec_x=dec_x, correlation_func=pearsonr, comp_func=np.max)
    correlation(real_x=x, rec_x=dec_x, correlation_func=root_mse, comp_func=np.min)

    xq = np.zeros(x.shape)
    index = 0
    for i in range(len(q)):
        xq[:, index:index + len(q[i])] = x[:, index:index + len(q[i])] @ q[i]
        index += len(q[i])

    print('#' * 30)
    print('Attack XQ')

    dec_x = one_side_attack(xq.T).T
    np.save(os.path.join('data', args.data + f'_xq_{args.seed}' + '.npy'), dec_x)
    correlation(real_x=x, rec_x=dec_x, correlation_func=pearsonr, comp_func=np.max)
    correlation(real_x=x, rec_x=dec_x, correlation_func=root_mse, comp_func=np.min)

    index = 0
    for i in range(len(q)):
        pxq[:, index:index + len(q[i])] = pxq[:, index:index + len(q[i])] @ q[i]
        index += len(q[i])

    print('#' * 30)
    print('Attack PXQ')

    dec_x = two_size_attack(pxq)
    np.save(os.path.join('data', args.data + f'_pxq_{args.seed}' + '.npy'), dec_x)
    correlation(real_x=x, rec_x=dec_x, correlation_func=pearsonr, comp_func=np.max)
    correlation(real_x=x, rec_x=dec_x, correlation_func=root_mse, comp_func=np.min)

    print('#' * 30)
    print('Correlation to random data')
    dec_x = np.random.random(x.shape)
    np.save(os.path.join('data', args.data + f'_random_{args.seed}' + '.npy'), dec_x)
    correlation(real_x=x, rec_x=dec_x, correlation_func=pearsonr, comp_func=np.max)
    correlation(real_x=x, rec_x=dec_x, correlation_func=root_mse, comp_func=np.min)
