attr="CEL"

for nchars in 8 10
do

for depth in 5
do

ho_rate=0.0
n_dictionaries=1
n_samples_per_dictionary=1000000
seed=0

job_name="dict_gen_${attr}_C${nchars}_ND${n_dictionaries}_NS${n_samples_per_dictionary}_D${depth}_S${seed}_HO${ho_rate}"

bash dict_gen.sh $attr $nchars $n_dictionaries $n_samples_per_dictionary $depth $seed $additional_args $ho_rate

done
done
