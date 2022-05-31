# script to train interactive bots in toy world
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools
import pdb
import random
import os
import numpy as np
from chatbots import Team
from dataloader import Dataloader
import options
from time import gmtime, strftime
import wandb


# read the command line options
options = options.read()
# ------------------------------------------------------------------------
# setup experiment and dataset
# ------------------------------------------------------------------------
data = Dataloader(options)
numInst = data.getInstCount()
useWandB = True
VERBOSE = True
if useWandB:
    wandb.init(project="lang-emerge", entity="nlp-2-a")

params = data.params
# append options from options to params
for key, value in options.items():
    params[key] = value

# ------------------------------------------------------------------------
# build agents, and setup optmizer
# ------------------------------------------------------------------------
team = Team(params)
if useWandB:
    wandb.watch(team.aBot)
    wandb.watch(team.qBot)
team.train()
optimizer = optim.Adam([{'params': team.aBot.parameters(),
                         'lr': params['learningRate']},
                        {'params': team.qBot.parameters(),
                         'lr': params['learningRate']}])
# ------------------------------------------------------------------------
# train agents
# ------------------------------------------------------------------------
# begin training
numIterPerEpoch = int(np.ceil(numInst['train']/params['batchSize']))
numIterPerEpoch = max(1, numIterPerEpoch)
count = 0
savePath = 'models/tasks_inter_%dH_%.4flr_%r_%d_%d.tar' %\
    (params['hiddenSize'], params['learningRate'], params['remember'],
     options['aOutVocab'], options['qOutVocab'])

matches = {}
accuracy = {}
bestAccuracy = 0
for iterId in range(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId)/numIterPerEpoch

    # get double attribute tasks
    if 'train' not in matches:
        batchImg, batchTask, batchLabels \
            = data.getBatch(params['batchSize'])
    else:
        batchImg, batchTask, batchLabels \
            = data.getBatchSpecial(params['batchSize'], matches['train'],
                                   params['negFraction'])

    # forward pass
    team.forward(Variable(batchImg), Variable(batchTask))
    # backward pass
    batchReward = team.backward(optimizer, batchLabels, epoch)

    # batch_size = batchLabels.size(0)
    # task_sizes = torch.Tensor([len(data.taskSelect[batchTask[batch_idx]]) for batch_idx in range(batch_size)]).int()
    # batchReward = team.backward(optimizer, batchLabels, epoch, task_sizes)

    # take a step by optimizer
    optimizer.step()
    # --------------------------------------------------------------------------
    # switch to evaluate
    team.evaluate()

    for dtype in ['train', 'test']:
        # get the entire batch
        # img: (batch x num_attrs) -> value for each attribute of image
        # task: (batch) -> idx of task in taskselect
        # labels: (batch x task_size) -> value for each attribute to be predicted
        img, task, labels = data.getCompleteData(dtype)
        task_sizes = torch.Tensor([len(data.taskSelect[task[batch_idx]]) for batch_idx in range(img.size(0))]).int()

        # evaluate on the train dataset, using greedy policy
        # guess: (task_size x batch) -> predicted values for task
        # guess_dist: (task_size x batch x 26?) -> ????
        # talk: (num_agents * rounds x batch) -> Messages sent between agents
        guess, guess_dist, talk = team.forward(Variable(img), Variable(task))

        # compute accuracy for color, shape, and both
        firstMatch = guess[0].data == labels[:, 0].long()
        secondMatch = guess[1].data == labels[:, 1].long()
        matches[dtype] = firstMatch & secondMatch
        accuracy[dtype] = 100*torch.sum(matches[dtype])\
            / float(matches[dtype].size(0))

        
        # compute accuracy for all attributes in entire batch
        # attr_matches: (task_size x batch )
        # attr_correct/attr_accuracy: scalar
        attr_matches = guess.data == labels.T.long()
        attr_correct = attr_matches.sum()
        attr_accuracy = attr_correct / task_sizes.sum()

        # # for each image/row in attr_matches, check if perfect 
        # # (i.e. all attrs correct)
        # rows_correct = []
        # for batch_idx in range(images.size(0)):
        #     # get current task size and image
        #     # current_task_size: scalar
        #     # current_attr_matches: (max_task_size)
        #     current_task_size = task_sizes[batch_idx]
        #     current_attr_matches = attr_matches[:, batch_idx]

        #     # only count predictions for current task
        #     # current_attr_matches: scalar
        #     current_attr_matches = attr_matches[:current_task_size].sum()
        #     # print(current_attr_matches, type(current_attr_matches))
        #     perfect_attr_predictions = current_attr_matches.long().equal( current_task_size.long())
        #     rows_correct.append(perfect_attr_predictions)

        # # rows_correct: (batch)
        # rows_correct = torch.Tensor(rows_correct)
        # # rows_accuracy: ratio of perfect image predictions to total images
        # rows_accuracy = sum(rows_correct) / batch_size
        # matches[dtype] = rows_correct
        # accuracy[dtype] = rows_accuracy
        # accuracy['attr_' + dtype] = attr_accuracy

    # switch to train
    team.train()

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100:
        break

    # save for every 5k epochs
    # if iterId > 0 and iterId % (10000*numIterPerEpoch) == 0:
    if iterId >= 0 and iterId % (10000*numIterPerEpoch) == 0:
        team.saveModel(savePath, optimizer, params)

    if iterId % 100 != 0:
        continue

    time = strftime("%a, %d %b %Y %X", gmtime())
    if useWandB:
        wandb.log({
            'time': time,
            'iter': iterId,
            'epoch': epoch,
            'totalReward': team.totalReward,
            'trainAccuracy': accuracy['train'],
            'testAccuracy': accuracy['test']
            # 'trainAttrAccuracy': accuracy['attr_train'],
            # 'testAttrAccuracy': accuracy['attr_test']
        })
    print('[%s][Iter: %d][Ep: %.2f][R: %.4f][Tr: %.2f Te: %.2f][AttrTr: %.2f AttrTe: %.2f]' %
          (time, iterId, epoch, team.totalReward,
           accuracy['train'], accuracy['test'],
           accuracy['train'], accuracy['test']))
        #    accuracy['attr_train'], accuracy['attr_test']))
# ------------------------------------------------------------------------
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime())
replaceWith = 'final_%s' % timeStamp
finalSavePath = savePath.replace('inter', replaceWith)
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer, params)
#------------------------------------------------------------------------

