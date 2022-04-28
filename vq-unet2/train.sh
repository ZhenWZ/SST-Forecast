python train_vq2.py --in_channels 16 \
    --out_channels 4 \
    --steps 1 \
    --skip 5 \
    --batch_size 4\
    --root '/home/featurize/data/Generate_Data_Step_0_496_264_20020601_20190409.mat' \
    --log_name '/home/featurize/log/vq2-16-4-skip5-fullscale-dim128-x4-loss80' \
    --checkpoint_name '/home/featurize/checkpoints/vq2-16-4-skip5-fullscale-loss1-dim128-x4-loss80'