DEVICES="1"

TRAIN_POST_PATH=../data/model_data/train/post.train
TRAIN_RESP_PATH=../data/model_data/train/resp.train
DEV_POST_PATH=../data/model_data/dev/post.dev
DEV_RESP_PATH=../data/model_data/dev/resp.dev
TRAIN_ID_PATH=../data/model_data/train/extra/resp_id.train
DEV_ID_PATH=../data/model_data/dev/extra/resp_id.dev
TRAIN_HIS_PATH=../data/model_data/train/his_resp_vec.train 
TRAIN_QUERY_PATH=../data/model_data/train/post_vec.train
DEV_HIS_PATH=../data/model_data/dev/his_resp_vec.dev
DEV_QUERY_PATH=../data/model_data/dev/post_vec.dev
SRC_VOCAB=../data/model_data/vocab/src_vocab_file
TGT_VOCAB=../data/model_data/vocab/tgt_vocab_file
USER_VOCAB=../data/model_data/vocab/tgt_userID_vocab_file

# Start training
python runModel.py \
        --device $DEVICES \
        --train_post_path $TRAIN_POST_PATH \
        --train_resp_path $TRAIN_RESP_PATH \
        --dev_post_path $DEV_POST_PATH \
        --dev_resp_path $DEV_RESP_PATH \
        --train_user_path $TRAIN_ID_PATH \
        --train_his_path $TRAIN_HIS_PATH \
        --train_query_path $TRAIN_QUERY_PATH \
        --dev_user_path $DEV_ID_PATH \
        --dev_his_path $DEV_HIS_PATH \
        --dev_query_path $DEV_QUERY_PATH \
        --src_vocab_file $SRC_VOCAB \
        --tgt_vocab_file $TGT_VOCAB \
        --user_vocab_file $USER_VOCAB \
        --bidirectional \
        --use_attn \
        --random_seed 2808 \
        # --resume
