
#!/bin/bash

N=1000

min_token = 200
length = 256

for remasking in low_confidence random; do
    for ratio in 0.5 0.75 1.0; do
        for strength in 1.0 2.0 5.0 10.0; do
            for strategy in bidirectional predict normal; do
                echo "Running generation with gen_length=${length}, strength=${strength}, min_token=${min_token}, ratio=${ratio}, remasking=${remasking}, strategy=${strategy}"
                python3 -m dmark.llada.only_gen --dataset sentence-transformers/eli5 --model GSAI-ML/LLaDA-8B-Instruct --num_samples ${N} --output_dir results --minimum_output_token ${min_token} --gen_length ${length} --steps ${length} --block_length 32 --temperature 0.0 --cfg_scale 0.0 --remasking ${remasking} --strategy ${strategy} --bitmap bitmap.bin --vocab_size 126464 --ratio ${ratio} --delta ${strength} --key 42
                echo "Completed gen_length=${length}, strength=${strength}, min_token=${min_token}, ratio=${ratio}, remasking=${remasking}, strategy=${strategy}"
                echo "-----------------------------------"
            done
        done
    done
done