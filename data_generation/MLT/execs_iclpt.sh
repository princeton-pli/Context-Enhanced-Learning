attr="ICLPT"

for nchars in 8 10
do

for depth in 5 6
do

n_dictionaries=1000000
n_samples_per_dictionary=1
seed=42
ho_rate=0.0

job_name="dict_gen_${attr}_C${nchars}_ND${n_dictionaries}_NS${n_samples_per_dictionary}_D${depth}_S${seed}_HO${ho_rate}"

bash dict_gen.sh $attr $nchars $n_dictionaries $n_samples_per_dictionary $depth $seed $ho_rate

done
done
