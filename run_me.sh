DATA_PATH='data/toy64_split_0.8.json'
# DATA_PATH='data/who_is_it.json'
VISIBLE_CUDA_DEVICES=0

# Train the agents.
python train.py -learningRate 0.01 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -dataset $DATA_PATH\
                -aOutVocab 4 -qOutVocab 3

# python train.py -learningRate 0.01 -hiddenSize 128 -batchSize 256 \
#                 -imgFeatSize 16 -embedSize 16 -numEpochs 25000\
#                 -dataset $DATA_PATH\
#                 -aOutVocab 4 -qOutVocab 2


# Test the agents (from a checkpoint) and visualize the dialogs.
CHECKPOINT="models/tasks_inter_100H_0.0100lr_False_4_3.tar"
# python test.py $CHECKPOINT
