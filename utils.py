import os
import pickle
import shutil

import numpy as np


def generate_orthogonal_matrix(
        n=1000, reuse=False, block_reduce=None, random_seed=None,
        only_inverse=False, file_name=None, memory_efficient=False, clear_cache=False
):
    orthogonal_matrix_cache_dir = 'tmp_orthogonal_matrices'

    if clear_cache:
        shutil.rmtree(orthogonal_matrix_cache_dir, ignore_errors=True)
        return True

    if random_seed:
        np.random.seed(random_seed)

    file_name = file_name or 'cached_matrix'

    if memory_efficient:
        assert reuse

    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    existing = set([e.split('.')[0] for e in os.listdir(orthogonal_matrix_cache_dir)])

    if block_reduce is not None:
        block_reduce = min(block_reduce, n)
        qs = [block_reduce] * int(n / block_reduce)
        if n % block_reduce != 0:
            qs[-1] += (n - np.sum(qs))
        q = []
        for i in range(len(qs)):
            sub_n = qs[i]
            piece_file_name = file_name + f'_piece{i}'
            if reuse and piece_file_name in existing:
                if memory_efficient:
                    tmp = os.path.join(orthogonal_matrix_cache_dir, piece_file_name)
                else:
                    with open(os.path.join(orthogonal_matrix_cache_dir, piece_file_name), 'rb') as f:
                        tmp = pickle.load(f)
            else:
                tmp = generate_orthogonal_matrix(
                    sub_n, reuse=False, block_reduce=None, only_inverse=only_inverse,
                    random_seed=random_seed if random_seed is None else (random_seed + i),
                )
                if reuse:
                    with open(os.path.join(orthogonal_matrix_cache_dir, piece_file_name), 'wb') as f:
                        pickle.dump(tmp, f, protocol=4)
                    if memory_efficient:
                        del tmp
                        tmp = os.path.join(orthogonal_matrix_cache_dir, piece_file_name)
            q.append(tmp)
    else:
        cache_file_name = os.path.join(orthogonal_matrix_cache_dir, file_name)
        if reuse and file_name in existing:
            if memory_efficient:
                q = cache_file_name
            else:
                with open(cache_file_name, 'rb') as f:
                    q = pickle.load(f)
        else:
            if not only_inverse:
                q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
            else:
                q = np.random.randn(n, n)
            if reuse:
                with open(cache_file_name, 'wb') as f:
                    pickle.dump(q, f, protocol=4)
                if memory_efficient:
                    del q
                    q = cache_file_name
    return q