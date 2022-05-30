# script to develop a toy example
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, json
import numpy as np
from chatbots import Team
from dataloader import Dataloader

import sys
sys.path.append('../')
from utilities import saveResultPage

useWandB = False
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

    # compute accuracy for first, second and both attributes
    firstMatch = preds[0].data == labels[:, 0].long()
    secondMatch = preds[1].data == labels[:, 1].long()
    matches = firstMatch & secondMatch
    atleastOne = firstMatch | secondMatch
    # # preds: (task_size x batch) -> predicted values for task
    # # guess_dist: (task_size x batch x 26?) -> ????
    # # talk: (num_agents * rounds x batch) -> Messages sent between agents
    # preds, guess_dist, talk = team.forward(Variable(images), Variable(tasks), True)
    # if VERBOSE:
    #     print('#1, preds:', preds.size())
    #     print('#1, idk:', len(guess_dist), guess_dist[0].size())
    #     print('#1, talk:', len(talk), talk[0].size())

    # # compute accuracy for first, second and both attributes
    # # iterate over task_size dim and compare preds with labels
    # # matches: (task_size x batch)
    # matches = preds.data == labels.T.long()
    # total_correct = matches.sum()

    # # evaluate accuracy, and number of rows correct 
    # # (i.e. correct for all attributes in a single image)
    # task_sizes = torch.Tensor([len(data.taskSelect[tasks[batch_idx]]) for batch_idx in range(batch_size)]).int()
    # total_accuracy = total_correct / (task_size * batch_size)
    # rows_correct = sum([matches[:task_sizes[batch_idx], batch_idx].sum() == task_sizes[batch_idx] for batch_idx in range(matches.size(1))])
    # if VERBOSE:
    #     print('#2, len task_sizes:', len(task_sizes))
    #     print('#2, matches:', matches.size())
    #     print('#2, total_accuracy:', total_accuracy)
    #     print('#2, rows_correct:', rows_correct)


    # # firstMatch = preds[0].data == labels[:, 0].long()
    # # secondMatch = preds[1].data == labels[:, 1].long()
    # # matches = firstMatch & secondMatch
    # # atleastOne = firstMatch | secondMatch

    # compute accuracy
    firstAcc = 100 * torch.mean(firstMatch.float())
    secondAcc = 100 * torch.mean(secondMatch.float())
    atleastAcc = 100 * torch.mean(atleastOne.float())
    accuracy = 100 * torch.mean(matches.float())
    print('\nOverall accuracy [%s]: %.2f (f: %.2f s: %.2f, atleast: %.2f)'\
                    % (dtype, accuracy, firstAcc, secondAcc, atleastAcc))

    # pretty print
    talk = data.reformatTalk(talk, preds, images, tasks, labels)
    if 'final' in loadPath:
        savePath = loadPath.replace('final', 'chatlog-'+dtype)
    elif 'inter' in loadPath:
        savePath = loadPath.replace('inter', 'chatlog-'+dtype)
    savePath = savePath.replace('tar', 'json')
    print('Saving conversations: %s' % savePath)
    with open(savePath, 'w') as fileId: json.dump(talk, fileId)
    saveResultPage(savePath)
