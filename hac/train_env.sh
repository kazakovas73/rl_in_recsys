mkdir -p output

# RL4RS environment

mkdir -p output/rl4rs/
mkdir -p output/rl4rs/env
mkdir -p output/rl4rs/env/log

data_path="dataset/ml-1m/"
output_path="output/ml-1m/"

for REG in 0.0003
do
    for LR in 0.001
    do
        python train_env.py\
            --model ML1MUserResponse\
            --reader ML1MDataReader\
            
            --train_file ${data_path}rl4rs_dataset_b_rl.csv\
            --val_file ${data_path}rl4rs_dataset_b_rl.csv\
            --item_meta_file ${data_path}item_info.csv\

            --user_meta_file$ ${data_path}
            
            --data_separator '@'\
            --meta_data_separator ' '\
            --loss 'bce'\
            --l2_coef ${REG}\
            --lr ${LR}\
            --epoch 2\
            --seed 19\
            --model_path ${output_path}env/rl4rs_user_env_lr${LR}_reg${REG}.model\
            --max_seq_len 50\
            --n_worker 4\
            --feature_dim 16\
            --hidden_dims 256\
            --attn_n_head 2\
            > ${output_path}env/log/rl4rs_user_env_lr${LR}_reg${REG}.model.log
    done
done