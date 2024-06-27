for learning_rates in .0001
do
  for betas in .0000001 .0000005 .000001 .000005 .00001
  do
    ./worker.sh $learning_rates $betas
  done
done