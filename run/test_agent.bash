name=lang_att_drop8_weight4_p3_5e-6_BS8

flag="--vlnbert prevalent

      --submit 1
      --test_only 0

      --train validlistener
      --load snap/lang_att_drop8_weight4_p3_5e-6_BS8/state_dict/best_val_unseen

      --features places365
      --maxAction 15
      --batchSize 16
      --TestbatchSize 16
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 r2r_src/train_debug.py $flag --name $name
