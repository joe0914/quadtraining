# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=200000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')

p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=5, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=40000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')

#p.add_argument('--pretrain_iters', type=int, default=100000, required=False, help='Number of pretrain iterations')
#p.add_argument('--counter_start', type=int, default=0, required=False, help='Defines the initial time for the curriculul training')
#p.add_argument('--counter_end', type=int, default=100000, required=False, help='Defines the linear step for curriculum training starting from the initial time')
#p.add_argument('--num_src_samples', type=int, default=20000, required=False, help='Number of source samples at each time step')
#p.add_argument('--num_target_samples', type=int, default=10000, required=False, help='Number of samples inside the target set')

p.add_argument('--minWith', type=str, default='maxVwithV0', required=False, choices=['none', 'zero', 'target', 'maxVwithV0'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--diffModel', action='store_true', default=False, required=False, help='Should we train the difference model instead.')
p.add_argument('--time_norm_mode', type=str, default='none', required=False, choices=['none', 'scale_ham', 'scale_PDE'])

p.add_argument('--periodic_boundary', action='store_true', default=True, required=False, help='Impose the periodic boundary condition.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=True, required=False, help='Pretrain dirichlet conditions')
p.add_argument('--adjust_relative_grads', action='store_true', default=False, required=False, help='Adjust relative gradients of the loss function.')
p.add_argument('--diffModel_mode', type=str, default='mode2', required=False, choices=['mode1', 'mode2'], help='BRS vs BRT computation')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

dataset = dataio.Q10DXY(numpoints=65000,
                                          pretrain=opt.pretrain, tMin=opt.tMin,
                                          tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                          pretrain_iters=opt.pretrain_iters,
                                          num_src_samples=opt.num_src_samples, periodic_boundary = opt.periodic_boundary, diffModel=opt.diffModel)


dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=6, out_features=1, type=opt.model, mode=opt.mode, #What is in feature vs out feature?
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model.cuda()
model.load_state_dict(torch.load('logs/qxy/checkpoints/model_final.pth'))
model.eval
#Define the loss
loss_fn = loss_functions.initialize_hji_Q10DXY(dataset, opt.minWith, opt.diffModel_mode)
alpha = dataset.alpha
beta = dataset.beta

def visualizeConverge(model):
  # Time values at which the function needs to be plotted
  times = [0., 0.25*opt.tMax, 0.5*opt.tMax, 0.75*opt.tMax, opt.tMax]
  num_times = len(times)


  slices_toplot = [{'th_x' : 0, 'o_x' : 2, 'pu_x': 0.5},
                   {'th_x' : 0, 'o_x' : 5, 'pu_x': 1}]
  num_slices = len(slices_toplot)

  #[state sequence: x, v_x, th_x, o_x, y, v_y, th_y, o_y, z, v_z]
  #Slices to be plotted
  fig = plt.figure(figsize=(5*num_times, 5*num_slices))

  # Get the meshgrid in the (x, v_x) coords. Rest are slices at regulated value
  sidelen = 100
  sidelen = (80, 80)
  mgrid_coords = dataio.get_mgrid(sidelen, dim = 2)

  # Start plotting the results
  for i in range(num_times):
    time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]
    for j in range(num_slices):
      coords = torch.cat((time_coords, mgrid_coords), dim=1) 
      
      #[theta_x, omega_x, y, v_y, theta_y, omega_y, z, v_z]
      thx_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['th_x'] - beta['th_x']) / alpha['th_x']
      ox_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['o_x'] - beta['o_x']) / alpha['o_x']
      pu_x_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['pu_x'] - beta['pu_x']) / alpha['pu_x']

      coords = torch.cat((coords,thx_coords,ox_coords, pu_x_coords), dim=1)
      #Input and output
      model_in = {'coords': coords.cuda()}
      model_out = model(model_in)['model_out']

      # Detatch model ouput and reshape
      model_out = model_out.detach().cpu().numpy()
      model_out = model_out.reshape(sidelen)

      # Unnormalize the value function
      model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean
      #Check change in the value function since last epoch checkpoint
      # e = <0.0001?
      model_out = np.sqrt(model_out)
      #model_out = np.max(model_out, axis = -1) # intersection over theta
      min_value = np.min(model_out)
      positive_mask = model_out > 0

      #This line didn't filter it out
      min_positive_value = np.min(model_out[positive_mask])
      print(min_positive_value)
      #Lowest converged value
      #model_out = np.where(model_out == min_positive_value, 1, 0)

      #Alternatively, add a buffer of model_out < min_pos_value + 0.1. Try 0.001, 0.005, 0.01. Excessive = 0.1, 0.0001, 0.0005
      model_out = np.where((model_out < min_positive_value+0.05) & (model_out > min_positive_value),1, 0)
      #model_out = (model_out > 0)*1

      ax = fig.add_subplot(num_times, num_slices, (j+1) + i*num_slices)
      ax.set_title('t = %0.2f' % (times[i]))
      s = ax.imshow(model_out.T, cmap='bwr', extent=(-alpha['x'], alpha['x'], -alpha['v_x'], alpha['v_x']), aspect=(alpha['x']/alpha['v_x']), vmin=-1., vmax=1.)
      fig.colorbar(s) 
      ax.set_aspect('equal')
    fig.savefig('qxy_minimum_converged_set_terminal_time.png', dpi=400)


def queryTEB(pu_x):
  sidelen = 100
  sidelen = (30, 30, 10, 10) #Grid size matters for accuracy
  mgrid_coords = dataio.get_mgrid(sidelen, dim = 4)


  # specific time (convergence, time = 5)
  time = 5
  time_coords = (torch.ones(mgrid_coords.shape[0], 1) * time)
  
  coords = torch.cat((time_coords, mgrid_coords), dim=1) 

  # specific pu_x pu_y
  pux_coords = (torch.ones(mgrid_coords.shape[0], 1) * pu_x - beta['pu_x'])/alpha['pu_x']
  coords = torch.cat((coords, pux_coords), dim=1)
  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']

  # Detatch model ouput and reshape
  model_out = model_out.detach().cpu().numpy()
  model_out = model_out.reshape(sidelen)
  # Unnormalize the value function
  model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 
  model_out = np.sqrt(model_out)
  positive_mask = model_out > 0

  #This line didn't filter it out
  TEB = np.min(model_out[positive_mask])
  #model_out = np.max(model_out, axis = -1) # intersection over theta
  return TEB

def queryVal(x, v_x, th_x, o_x, virt_v):
  coords = torch.ones(1,6)
  # specific time (convergence, time = 5)
  time = 5
  # specific pu_x pu_y
  pu_x = virt_v
  coords[:,0] = coords[:,0] * time

  coords[:,1] = coords[:,1] * (x - beta['x'])/alpha['x']
  coords[:,2] = coords[:,2] * (v_x - beta['v_x'])/alpha['v_x']
  coords[:,3] = coords[:,3] * (th_x - beta['th_x'])/alpha['th_x']
  coords[:,4] = coords[:,4] * (o_x - beta['o_x'])/alpha['o_x']

  coords[:,5] = coords[:,5] * (pu_x - beta['pu_x'])/alpha['pu_x']

  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']
  # Detatch model ouput and reshape
  model_out = model_out.detach().cpu().numpy()
  val = model_out[0][0]

  return val

def computeGradient(x, v_x, th_x, o_x, virt_v):
  delta = 0.01 #Delta
  gradient_u = (queryVal(x, v_x, th_x, o_x+ delta, virt_v) - queryVal(x, v_x, th_x, o_x- delta, virt_v)) / (2 * delta)

  return gradient_u

def optCtrl(gradient):
  xy_bound = [-(20/180)*math.pi, (20/180)*math.pi]
  ctrl = (gradient>=0)*xy_bound[0] + (gradient<0)*xy_bound[1]

  return np.array([ctrl])

visualizeConverge(model)
print(queryTEB(0.5))
print(queryTEB(1))
print(queryTEB(2))
#print(optCtrl(computeGradient(x=-1, v_x=2, th_x=2, o_x= 2, virt_v=0.5)))

