batchsizes=( 4 8 16 32 64 128 256 )
model_sizes=( 1.3 2.7 6.7 13 )

for batchsize in "${batchsizes[@]}" ; do
    for model_size in "${model_sizes[@]}" ; do
        ./gpg configs/OPT_${model_size}/${batchsize}-profile-InputPF.config >> output-InputPF.log && ./gpg configs/OPT_${model_size}/${batchsize}-profile-PF.config >> output-PF.log && ./gpg configs/OPT_${model_size}/${batchsize}-profile.config >> output.log
    done
done

for batchsize in "${batchsizes[@]}" ; do
    for model_size in "${model_sizes[@]}" ; do
        ./gpg configs/OPT_${model_size}/${batchsize}-sim_lru.config >> output.log
    done
done