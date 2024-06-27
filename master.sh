for learning_rates in .000001 .00001 .00005 .0001
do
  for betas in .0000001 .000001 .00001
  do
    ./worker.sh $learning_rates $betas
  done
done