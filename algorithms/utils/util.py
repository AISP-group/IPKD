import copy
import numpy as np

import torch
import torch.nn as nn

import random
import os
import shutil

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clearOldLogs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def rename_points_descriptions():
    updated_point_names = []
    '''
    #GFR env
    updated_point_names = ['top', 'bottom','left','right'
                           'top-left', 'top-right', 'bottom-left', 'bottom-right',
                           'short pass', 'high pass', 'long pass', 'shot',
                           'keeper rush', 'sliding', 'dribble', 'stop-dribble',
                           'sprint','stop-moving','stop-sprint','do-nothing']
    '''
    '''SMAC env'''
    '''
    updated_point_names = ['no operation', 'stop',
                            'move north', 'move sorth','move east', 'move west',
                            'attack enemy1', 'attack enemy2','attack enemy3', 'attack enemy4', 'attack enemy5','attack enemy6', 'attack enemy7', 'attack enemy8', 'attack enemy9']
    '''
    '''['no operation', 'stop',
                               'move north', 'move sorth',
                               'move east', 'move west', 'attack enemy', 'kiting',
                               'alternating fire', 'wall off', 'positioning']'''
    '''
    updated_point_names = ['no operation', 'stop',
                            'move north', 'move sorth','move east', 'move west',
                            'attack enemy1', 'attack enemy2','attack enemy3', 'attack enemy4', 'attack enemy5']
    '''
    # updated_point_names = ['root_x forward', 'root_y forward','root_z forward',
    #                        'bfoot and bshin connection', 'bshin and bthigh connection', 'bthigh and fthigh connection', 'fthigh and fshin connection',
    #                        'fshin and ffoot connection']
    updated_point_names = ['forward', 'balance','fast',
                           'joint coordination', 'control speed', 'No wrestling', 'smooth',
                           'stable']
    return updated_point_names

def rename_state_descriptions():
    updated_state_names = []

    '''
    #GFR env
    updated_state_names = ['top', 'bottom','left','right'
                           'top-left', 'top-right', 'bottom-left', 'bottom-right',
                           'short pass', 'high pass', 'long pass', 'shot',
                           'keeper rush', 'sliding', 'dribble', 'stop-dribble',
                           'sprint','stop-moving','stop-sprint','do-nothing']
    '''
    #SMAC env
    '''
    updated_state_names = ['Control stalkers to attack the zealots simultaneously',
    'After dealing damage, retreat slightly to avoid the zealots counter-attack',
    'Repeat this process to slowly whittle down the zealots health',
    'move fast to keep oneself safe first',
    'if safe then focus fire to attack enemy',
    'move keep oneself safe then attack enemy']
    '''
    '''
    updated_state_names = ['move to avoid the zealots counter-attack',
                           'move fast to keep oneself safe first',
                           'if safe then focus fire to attack enemy',
                           'attack  enemy with low health']
    '''
    #27m_vs_30m
    '''
    updated_state_names = ['move away from each other',
                           'place the center towards the densest part of enemy',
                           'sides behind to guard against flanking maneuvers',
                           'focus fire on the most vulnerable','focus fire on threatening enemies',
                           'focus fire on low health','focus fire on standing alone','focus fire on those dealing high damage',
                           'eliminating low health','eliminating high damage enemys']
    '''

    #3s5z_vs_3s6z
    #updated_state_names1 = ['Stalkers','Zealots']
    #updated_state_names2 = ['Stalkers Utilize their long range to deal damage outside the enemy attack range','Stalkers can lure the enemy fire and then pull away to allow zealots to counterattack','Zealots absorb damage in the front row and deal melee damage',    'Position the zealots in the front to attract the fire of the enemy stalkers and zealots','creating an output environment for the stalkers in the rear']
    '''
    updated_state_names = ['Utilize their long range to deal damage', 'outside the enemy attack range',
                            'lure the enemy fire ','pull away','counterattack',
                            'absorb damage in the front row','deal melee damage',
                            'position in the front','attract the fire',
                            'creating an output environment in the rear']
    '''
    
    #mujoco
    '''
    updated_state_names = ['Prioritize propelling the center of mass forward','Implement coordinated actuation between forelimbs and hindlimbs',
                          'Implement coordinated actuation between forelimbs and hindlimbs',
                          ' extending the leading limb','the trailing limb should generate propulsive thrust through ground reaction forces',
                          'Control joint motion ranges to maintain postural equilibrium and dynamic stability',
                          'Avoid compromised stability or excessive energy expenditure caused by excessive actuation']
    '''
    updated_state_names = ['propelling the center of mass forward','maintain postural equilibrium',
                          'maintain dynamic stability','Balance first and then move forward','reduce punishment',
                          'forward','thrust the ground',
                          'save energy','reduce motor torque',
                          'Avoid excessive energy expenditure']
 
    return updated_state_names