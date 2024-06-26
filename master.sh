for learning_rates in .0001
do
  for latent_dims in 512
  do
    for betas in 0 .000001 .00001 1
    do
      ./worker.sh $learning_rates $latent_dims $betas
    done
  done
done