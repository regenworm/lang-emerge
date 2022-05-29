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
VERBOSE = True
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
    # images: (batch x num_attrs) -> value for each attribute of image
    # tasks: (batch) -> idx of task in taskselect
    # labels: (batch x task_size) -> value for each attribute to be predicted
    images, tasks, labels = data.getCompleteData(dtype)
    task_size = labels.size(1)
    batch_size = labels.size(0)
    if VERBOSE:
        print('#0, images:', images.size())
        print('#0, tasks:', tasks.size())
        print('#0, labels:', labels.size())

    # forward pass
    # preds: (task_size x batch) -> predicted values for task
    # guess_dist: (task_size x batch x 26?) -> ????
    # talk: (num_agents * rounds x batch) -> Messages sent between agents
    preds, guess_dist, talk = team.forward(Variable(images), Variable(tasks), True)
    if VERBOSE:
        print('#1, preds:', len(preds), preds[0].size())
        print('#1, idk:', len(guess_dist), guess_dist[0].size())
        print('#1, talk:', len(talk), talk[0].size())

    # compute accuracy for first, second and both attributes
    # TODO: Adjust code so pairwise tasks are not the only tasks possible

    # iterate over task_size dim and compare preds with labels
    total_correct = []
    for idx, p in enumerate(preds):
        # match: (batch)
        match = (p.data  == labels[:, idx].long())
        total_correct.append( match.int())
        # print('match size', match.size(), match.sum())

    # evaluate accuracy, and number of rows correct 
    # (i.e. correct for all attributes in a single image)
    # rows_totals: (batch)
    total_accuracy = sum(sum(total_correct)) / (task_size * batch_size)
    row_totals = sum(total_correct)
    rows_correct = sum([rt == len(data.taskSelect[tasks[idx]]) for idx, rt in enumerate(row_totals)])
    if VERBOSE:
        print('#2, total_accuracy:', total_accuracy)
        print('#2, row_totals:', row_totals.size())
        print('#2, rows_correct:', rows_correct)


    # firstMatch = preds[0].data == labels[:, 0].long()
    # secondMatch = preds[1].data == labels[:, 1].long()
    # matches = firstMatch & secondMatch
    # atleastOne = firstMatch | secondMatch

    # # compute accuracy
    # firstAcc = 100 * torch.mean(firstMatch.float())
    # secondAcc = 100 * torch.mean(secondMatch.float())
    # atleastAcc = 100 * torch.mean(atleastOne.float())
    # accuracy = 100 * torch.mean(matches.float())
    # print('\nOverall accuracy [%s]: %.2f (f: %.2f s: %.2f, atleast: %.2f)'\
    #                 % (dtype, accuracy, firstAcc, secondAcc, atleastAcc))

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
