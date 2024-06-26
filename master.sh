for learning_rates in .0000001 .000001 .00001 
do
  for latent_dims in 128 256 512
  do
    for betas in 0 .00001 .0001 .001 .01 .1 1
    do
      ./worker.sh $learning_rates $latent_dims $betas
    done
  done
done