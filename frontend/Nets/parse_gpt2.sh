#!/bin/bash

# echo "Parsing models to parsed_models/ ..."
# batchsizes=( 4 6 8 12 16 20 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096 6144 8192 )

# for batchsize in "${batchsizes[@]}" ; do
#     echo "Parsing size: ${batchsize} "
#     python3 model_parser.py "gpt2-10-b${batchsize}.json"
# done

# seq_lengths=( 66 ) # 96 128 256 384 512 768 1024 1536 2048 4096 8192)
model_sizes=( "1.3" "2.7" "6.7" "13" )

for model_size in "${model_sizes[@]}" ; do
    # for seq_length in "${seq_lengths[@]}" ; do
        echo "Parsing size: ${model_size} "
        python3 parse_transformer_model.py "opt-${model_size}.json" > OPT_${model_size}.txt
    # done
done

# echo "Parsing models to TRxpr_xxx.json ..."

# for batchsize in "${batchsizes[@]}" ; do
#     echo "Parsing size: ${batchsize} "
#     python3 model_parser.py "gpt2-10-b${batchsize}.json" --parsed
# done

# batchsizes=( 2-l 4-l )
# model_sizes=( "13" )

# for model_size in "${model_sizes[@]}" ; do
#     for batchsize in "${batchsizes[@]}" ; do
#         echo "Parsing size: ${batchsize} "
#         python3 parse_transformer_model.py "opt-${model_size}-b${batchsize}.json"
#     done
# done 