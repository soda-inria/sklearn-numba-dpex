rm benchmark_results.json

for work_group_size in 64 128 256 512 1024
do
  for sub_group_size in 4 8 16 32 64
  do 
    for arithmetic_intensity_multiplier_X in 1 2 4 8 16 32
    do
      for arithmetic_intensity_multiplier_Y in 1 2 4 8 16 32
      do
        for private_Y_t_sliding_window_width in 1 2 4 8 16 32
        do
          python ./matmul_benchmark_run.py --work-group-size $work_group_size --sub-group-size $sub_group_size --arithmetic-intensity-multiplier-X $arithmetic_intensity_multiplier_X --arithmetic-intensity-multiplier-Y $arithmetic_intensity_multiplier_Y --private-Y-t-sliding-window-width $private_Y_t_sliding_window_width
        done
      done
    done
  done
done

