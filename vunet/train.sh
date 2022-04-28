python train.py --in_channels 16 \
    --out_channels 4 \
    --steps 1 \
    --skip 5 \
    --batch_size 4\
    --root '/home/featurize/data/Generate_Data_Step_0_496_264_20020601_20190409.mat' \
    --log_name '/home/featurize/log/version-16-4-skip5-fullscale' \
    --checkpoint_name '/home/featurize/work/SST-Forecast/vunet/checkpoints/version-16-4-skip5-fullscale-loss1/checkpoint-1' 