#!/bin/bash

data_root=$HOME/data/crossbeam/logic_synthesis_10hr
output_root=$HOME/data/crossbeam/logic_synthesis_dedup

if [ ! -e $output_root ];
then
	mkdir -p $output_root
fi

#python clean_logic.py \
#  $data_root/train-tasks.pkl \
#  $output_root/train-tasks.pkl \

python make_shards.py \
  $output_root
