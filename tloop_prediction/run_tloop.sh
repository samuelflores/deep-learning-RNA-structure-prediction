export DATA_PATH=../data_generation/training_data/non_homology_reduced
export OUTPUT_PATH=./results/DNABERT2/alvis/tetraloop
export TRAIN_PATH=.
export LR=3e-5

# Permute through the following settings:
# RNG seed (1-20)
# fragment length
# train:test ratio (80:10:10 original paper, try smth else)
# change bases from RNA to DNA

for seed in 1 2 3 #1 2 3 4 5 6 7 8 9 10
do
    for task in clusters #gnra clusters tloop gnravall # TODO run for all clusters
    do
        for fragment_length in 8 10 12 14 16 18 20 22 24 #8 10 12 14 16 18 20 22 24
        do
            for train_ratio in 80
            do
                for nucleotides in T U # T U
                do
                    # Training use DataParallel
                    python ${TRAIN_PATH}/train_dnabert2.py \
                        --model_name_or_path zhihan1996/DNABERT-2-117M \
                        --data_path  ${DATA_PATH}/${task}_${fragment_length}_${train_ratio}_${nucleotides} \
                        --kmer -1 \
                        --run_name DNABERT2_lr${LR}_task${task}_fragment${fragment_length}_trainratio${train_ratio}_nt${nucleotides}_seed${seed}/ \
                        --model_max_length $((${fragment_length}/4)) \
                        --per_device_train_batch_size 8 \
                        --per_device_eval_batch_size 16 \
                        --gradient_accumulation_steps 1 \
                        --learning_rate ${LR} \
                        --num_train_epochs 5 \
                        --fp16 \
                        --save_steps 200 \
                        --output_dir ${OUTPUT_PATH}/${task}_${fragment_length}_${train_ratio}_${nucleotides}_${seed}/ \
                        --evaluation_strategy steps \
                        --eval_steps 200 \
                        --warmup_steps 50 \
                        --logging_steps 100 \
                        --overwrite_output_dir True \
                        --log_level info \
                        --find_unused_parameters False \
                        --seed ${seed}
                done
            done
        done
    done
done
