name=test

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0

      --train auglistener
      --num_process 3

      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 5e-6
      --iters 100000
      --optim adamW

      --mlWeight 0.20
      --distance_weight 6.0
      --distance_weight_c 2.0
      --drop_rate 0.5

      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 r2r_src/train.py $flag --name $name
