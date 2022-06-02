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
    wandb.init(project="lang-emerge", entity="nlp-2-a", tags=['train', 'baseline', 'test_variable_task'])
    artifact_predictions = wandb.Artifact('predictions', type='results')
    artifact_talks = wandb.Artifact('talks', type='results')
    artifact_gt = wandb.Artifact('gt', type='results')

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
        # get negFraction*batch_size amount of batch to be images that were previously
        # predicted wrong.
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
        # match_iter: (task_size x batch)
        match_iter = []
        for current_task in range(guess.size(0)):
            m = guess[current_task].data == labels[:, current_task]
            match_iter.append(m)
        # perfect_matches: (batch)
        perfect_matches = [match_iter[idx] & match_iter[idx+1] for idx in range(len(match_iter)-1)]

        if VERBOSE:
            # check if results are same as method used for hardcoded 2 tasks
            firstMatch = guess[0].data == labels[:, 0].long()
            secondMatch = guess[1].data == labels[:, 1].long()
            same = ((firstMatch & secondMatch) == perfect_matches[0]).all()
            if not same:
                print('NOT SAME!', same)

        matches[dtype] = perfect_matches[-1]
        accuracy[dtype] = 100*torch.sum(matches[dtype])\
            / float(matches[dtype].size(0))
        # sum over task dimension, and take mean over batch
        # task_size * batch_size
        total_num_attrs = guess.size(0) * guess.size(1)
        accuracy['attr_' + dtype] = (sum(match_iter).float().sum() / total_num_attrs) *100
        match_iter = torch.stack(match_iter) 
        # sum match iter over batch size
        accuracy[dtype+ '_pertask'] = match_iter.float().mean(1) * 100

        if useWandB & (dtype == task) & (iterId % 100 == 0):
            artifact_predictions.add(guess, iterId)
            artifact_talks.add(talk, iterId)
            artifact_gt.add(labels, iterId)

    # switch to train
    team.train()

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
        'testAccuracy': accuracy['test'],
        'trainAttrAccuracy': accuracy['attr_train'],
        'testAttrAccuracy': accuracy['attr_test'],
        'testAccuracyPerAttr': wandb.Histogram(accuracy['test_pertask'].cpu()),
        'trainAccuracyPerAttr': wandb.Histogram(accuracy['train_pertask'].cpu()),
    })
    print('[%s][Iter: %d][Ep: %.2f][R: %.4f][Tr: %.2f Te: %.2f][AttrTr: %.2f AttrTe: %.2f]' %
          (time, iterId, epoch, team.totalReward,
           accuracy['train'], accuracy['test'],
        #    accuracy['train'], accuracy['test']))
           accuracy['attr_train'], accuracy['attr_test']))

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100:
        break

    if accuracy['test'] > 95:
        break
# ------------------------------------------------------------------------
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime())
replaceWith = 'final_%s' % timeStamp
finalSavePath = savePath.replace('inter', replaceWith)
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer, params)
#------------------------------------------------------------------------

