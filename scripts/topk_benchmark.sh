rm benchmark_results.json

for work_group_size in 64 128 256 512 1024
do
  for sub_group_size in 4 8 16 32 64
  do
          python ./topk_benchmark_run.py --work-group-size $work_group_size --sub-group-size $sub_group_size
  done
done
