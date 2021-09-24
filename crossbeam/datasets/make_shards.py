import os
import pickle as cp
import sys
from tqdm import tqdm


if __name__ == '__main__':
  data_folder = sys.argv[1]
  with open(os.path.join(data_folder, 'train-tasks.pkl'), 'rb') as f:
    all_tasks = cp.load(f)
    print('# tasks', len(all_tasks))
  shard_size = 10000
  for sid, i in tqdm(enumerate(range(0, len(all_tasks), shard_size))):
    with open(os.path.join(data_folder, 'train-%05d.pkl' % sid), 'wb') as fout:
      cp.dump(all_tasks[i : i + shard_size], fout, cp.HIGHEST_PROTOCOL)

