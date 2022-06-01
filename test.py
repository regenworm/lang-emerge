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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loaded = torch.load(loadPath, map_location=device)

#------------------------------------------------------------------------
# build dataset, load agents
#------------------------------------------------------------------------
params = loaded['params']
params['useGPU'] = torch.cuda.is_available()
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

    # compute accuracy for color, shape, and both
    # match_iter: (task_size x batch)
    match_iter = []
    for current_task in range(preds.size(0)):
        m = preds[current_task].data == labels[:, current_task]
        match_iter.append(m)
    # perfect_matches: (batch)
    perfect_matches = [match_iter[idx] & match_iter[idx+1] for idx in range(len(match_iter)-1)]

    if VERBOSE:
        # check if results are same as method used for hardcoded 2 tasks
        firstMatch = preds[0].data == labels[:, 0].long()
        secondMatch = preds[1].data == labels[:, 1].long()
        same = ((firstMatch & secondMatch) == perfect_matches[0]).all()
        if not same:
            print('NOT SAME!', same)

    # compute accuracy
    total_accuracy = 100*torch.sum(perfect_matches[-1])\
        / float(perfect_matches[-1].size(0))

    # sum over task dimension, and take mean over batch
    # task_size * batch_size
    total_num_attrs = preds.size(0) * preds.size(1)
    attr_accuracy = (sum(match_iter).float().sum() / total_num_attrs) *100
    print('\nPer attribute accuracy: ', [(torch.mean(attr.float())*100).item() for attr in match_iter])
    print(f'\nTotal accuracy: {total_accuracy}, Attribute Accuracy: {attr_accuracy}')

    # pretty print
    # talk: (batch) -> { 
    #   image: (num_attr), 
    #   gt: (task_len), 
    #   task: (num_tasks), 
    #   pred: (num_tasks), 
    #   chat: (num_rounds * num_agents) 
    # }
    talk = data.reformatTalk(talk, preds, images, tasks, labels)
    if 'final' in loadPath:
        savePath = loadPath.replace('final', 'chatlog-'+dtype)
    elif 'inter' in loadPath:
        savePath = loadPath.replace('inter', 'chatlog-'+dtype)
    savePath = savePath.replace('tar', 'json')
    print('Saving conversations: %s' % savePath)
    with open(savePath, 'w') as fileId: json.dump(talk, fileId)
    saveResultPage(savePath)
