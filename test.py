# script to develop a toy example
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import wandb

import itertools, pdb, random, json
import numpy as np
from chatbots import Team
from dataloader import Dataloader

import sys
sys.path.append('../')
from utilities import saveResultPage

useWandB = True
#------------------------------------------------------------------------
# load experiment and model
#------------------------------------------------------------------------
if len(sys.argv) < 2:
    print('Wrong usage:')
    print('python test.py <modelPath>')
    sys.exit(0)

# load and compute on test
loadPath = sys.argv[1]
print('Loading model from: %s' % loadPath)
loaded = torch.load(loadPath)
if useWandB:
    wandb.init(project="lang-emerge", entity="nlp-2-a", tags=['test'])

#------------------------------------------------------------------------
# build dataset, load agents
#------------------------------------------------------------------------
params = loaded['params']
data = Dataloader(params)

team = Team(params)
team.loadModel(loaded)
team.evaluate()
#------------------------------------------------------------------------
# test agents
#------------------------------------------------------------------------
dtypes = ['train', 'test']
for dtype in dtypes:
    # evaluate on the train dataset, using greedy policy
    images, tasks, labels = data.getCompleteData(dtype)
    # forward pass
    preds, _, talk = team.forward(Variable(images), Variable(tasks), True)
    print('talk', len(talk))
    # compute accuracy for first, second and both attributes
    # TODO: Adjust code so pairwise tasks are not the only tasks possible
    firstMatch = preds[0].data == labels[:, 0].long()
    secondMatch = preds[1].data == labels[:, 1].long()
    matches = firstMatch & secondMatch
    atleastOne = firstMatch | secondMatch

    # compute accuracy
    firstAcc = 100 * torch.mean(firstMatch.float())
    secondAcc = 100 * torch.mean(secondMatch.float())
    atleastAcc = 100 * torch.mean(atleastOne.float())
    accuracy = 100 * torch.mean(matches.float())
    print('\nOverall accuracy [%s]: %.2f (f: %.2f s: %.2f, atleast: %.2f)'\
                    % (dtype, accuracy, firstAcc, secondAcc, atleastAcc))

    # pretty print
    talk = data.reformatTalk(talk, preds, images, tasks, labels)
    if useWandB:
        table = wandb.Table(columns=list(talk[0].keys()))
        for row in talk:
            table.add_data(*list(row.values()))
            # print('hmm', fail)
        wandb.log({'test_table': table})
    # print('talk pretty', talk[0].keys())
    if 'final' in loadPath:
        savePath = loadPath.replace('final', 'chatlog-'+dtype)
    elif 'inter' in loadPath:
        savePath = loadPath.replace('inter', 'chatlog-'+dtype)
    savePath = savePath.replace('tar', 'json')
    print('Saving conversations: %s' % savePath)
    with open(savePath, 'w') as fileId: json.dump(talk, fileId)
    saveResultPage(savePath)
