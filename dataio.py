import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import diff_operators

import utils
import pickle


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)
    if dim == 1:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2],  :sidelen[3]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[3] - 1)
    elif dim == 5:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2],  :sidelen[3], :sidelen[4]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[3] - 1)
        pixel_coords[..., 4] = pixel_coords[..., 4] / (sidelen[4] - 1)
    elif dim == 6:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2],  :sidelen[3], :sidelen[4], :sidelen[5]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[3] - 1)
        pixel_coords[..., 4] = pixel_coords[..., 4] / (sidelen[4] - 1)
        pixel_coords[..., 5] = pixel_coords[..., 5] / (sidelen[5] - 1)
    elif dim == 10:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2],  :sidelen[3], :sidelen[4], :sidelen[5],  :sidelen[6], :sidelen[7], :sidelen[8],  :sidelen[9]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[3] - 1)
        pixel_coords[..., 4] = pixel_coords[..., 4] / (sidelen[4] - 1)
        pixel_coords[..., 5] = pixel_coords[..., 5] / (sidelen[5] - 1)
        pixel_coords[..., 6] = pixel_coords[..., 6] / (sidelen[6] - 1)
        pixel_coords[..., 7] = pixel_coords[..., 7] / (sidelen[7] - 1)
        pixel_coords[..., 8] = pixel_coords[..., 8] / (sidelen[8] - 1)
        pixel_coords[..., 9] = pixel_coords[..., 9] / (sidelen[9] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def to_uint8(x):
    return (255. * x).astype(np.uint8)

def to_numpy(x):
    return x.detach().cpu().numpy()

def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

class Quadcopter10DP3D(Dataset):
    #     Constructor for a 10D quadrotor
    #  
    #       Dynamics of the 10D Quadrotor
    #           \dot x_1 = x_2 - d_1
    #           \dot x_2 = g * tan(x_3)
    #           \dot x_3 = -d1 * x_3 + x_4
    #           \dot x_4 = -d0 * x_3 + n0 * u1
    #           \dot x_5 = x_6 - d_2
    #           \dot x_6 = g * tan(x_7)
    #           \dot x_7 = -d1 * x_7 + x_8
    #           \dot x_8 = -d0 * x_7 + n0 * u2
    #           \dot x_9 = x_10 - d_3
    #           \dot x_10 = kT * u3
    #                uMin <= [u1; u2; u3] <= uMax
    #                dMin <= [d1; d2; d3] <= dMax
    def __init__(self, numpoints,
                 tMin=0.0, tMax=5, 
        pretrain=False, counter_start=0, counter_end=100e3, pretrain_iters=2000, 
        num_src_samples=1000, periodic_boundary=False, diffModel=False):
        super().__init__()
        torch.manual_seed(0)
        self.pretrain = pretrain
        self.periodic_boundary = periodic_boundary
        self.diffModel = diffModel
        #self.sample_inside_target = True
        self.numpoints = numpoints

        #Dynamics parameters
        self.num_states = 13

        #Time parameters
        self.tMax = tMax
        self.tMin = tMin


        # Normalization for states and time
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x, v_x, theta_x, omega_x, y, v_y, theta_y, omega_y, z, v_z]
        self.alpha = {}
        self.beta = {}

        self.alpha['x'] = 5.0 # x in [-5, 5] range
        self.alpha['v_x'] = 2.0 # v_x in [1, 5] range
        self.alpha['th_x'] = 1.0*math.pi #[-pi, pi] range
        self.alpha['o_x'] =  2.0 #o_y in [1, 5] range

        self.alpha['y'] = 5.0 # y in [-5, 5] range
        self.alpha['v_y'] = 2.0 # v_y in [1, 5] range
        self.alpha['th_y'] = 1.0*math.pi #[-pi, pi] range
        self.alpha['o_y'] =  2.0 #o_y in [1, 5] range
        
        self.alpha['z'] = 5.0 # z in [-5, 5] range
        self.alpha['v_z'] = 2.0 # v_z in [1, 5] range

        self.alpha['time'] = 1 #time is in [0, 1] range

        self.beta['x'] = 0
        self.beta['v_x'] = 3.0
        self.beta['th_x'] = 0
        self.beta['o_x'] =  3.0

        self.beta['y'] = 0
        self.beta['v_y'] = 3.0
        self.beta['th_y'] = 0
        self.beta['o_y'] =  3.0
        
        self.beta['z'] = 0
        self.beta['v_z'] = 3.0

        self.alpha['pu_x'] = 2.5 #planner control is in [0.5, 5] range
        self.alpha['pu_y'] = 2.5 #planner control is in [0.5, 5] range
        self.alpha['pu_z'] = 2.5

        self.beta['pu_x'] = 2.5
        self.beta['pu_y'] = 2.5
        self.beta['pu_z'] = 2.5
        # Normalization for the value function
        self.norm_to = 0.02 #Ultimate range after normalization [-0.02, 0.02]
        self.mean = 4.30  #Max val of l(x), and min val of l(x), find mean & var
        self.var = 4.195 # 
        #( ( v - mean ) / var  ) * norm_to
        #Pick mean and var so that this output is between [-1, 1]

        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2
        self.N_inside_target_samples = num_src_samples*5

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.zeroTensor = torch.tensor(0.).cuda()

    def __len__(self):
        return 1
    
    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[..., 0] = state_coords_unnormalized[..., 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[..., 1] = state_coords_unnormalized[..., 1] * self.alpha['v_x'] + self.beta['v_x']
        state_coords_unnormalized[..., 2] = state_coords_unnormalized[..., 2] * self.alpha['th_x'] + self.beta['th_x']
        state_coords_unnormalized[..., 3] = state_coords_unnormalized[..., 3] * self.alpha['o_x'] + self.beta['o_x']

        state_coords_unnormalized[..., 4] = state_coords_unnormalized[..., 4] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[..., 5] = state_coords_unnormalized[..., 5] * self.alpha['v_y'] + self.beta['v_y']
        state_coords_unnormalized[..., 6] = state_coords_unnormalized[..., 6] * self.alpha['th_y'] + self.beta['th_y']
        state_coords_unnormalized[..., 7] = state_coords_unnormalized[..., 7] * self.alpha['o_y'] + self.beta['o_y']

        state_coords_unnormalized[..., 8] = state_coords_unnormalized[..., 8] * self.alpha['z'] + self.beta['z']
        state_coords_unnormalized[..., 9] = state_coords_unnormalized[..., 9] * self.alpha['v_z'] + self.beta['v_z']
        
        state_coords_unnormalized[..., 10] = state_coords_unnormalized[..., 10] * self.alpha['pu_x'] + self.beta['pu_x']
        state_coords_unnormalized[..., 11] = state_coords_unnormalized[..., 11] * self.alpha['pu_y'] + self.beta['pu_y']
        state_coords_unnormalized[..., 12] = state_coords_unnormalized[..., 12] * self.alpha['pu_z'] + self.beta['pu_z']


        # positional states from start
        boundary_values = torch.norm(state_coords_unnormalized[..., [0,4,8]], dim=-1, keepdim=True)
        return boundary_values
    
    def compute_overall_ham(self, x, dudx):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x'] #p1
        dudx[..., 1] = dudx[..., 1] / alpha['v_x'] #p2
        dudx[..., 2] = dudx[..., 2] / alpha['th_x'] #p3
        dudx[..., 3] = dudx[..., 3] / alpha['o_x'] #p4

        dudx[..., 4] = dudx[..., 4] / alpha['y'] #p5
        dudx[..., 5] = dudx[..., 5] / alpha['v_y'] #p6
        dudx[..., 6] = dudx[..., 6] / alpha['th_y'] #p7
        dudx[..., 7] = dudx[..., 7] / alpha['o_y'] #p8

        dudx[..., 8] = dudx[..., 8] / alpha['z'] #p9
        dudx[..., 9] = dudx[..., 9] / alpha['v_z'] #p10
        
        dudx[..., 10] = dudx[..., 10] / alpha['pu_x'] #p11
        dudx[..., 11] = dudx[..., 11] / alpha['pu_y'] #p12
        dudx[..., 12] = dudx[..., 12] / alpha['pu_z'] #p13
    
        # Scale the states appropriately.
        x_u = x * 1.0
        x_u[..., 1] = x_u[..., 1] * alpha['x'] + beta['x'] #x1
        x_u[..., 2] = x_u[..., 2] * alpha['v_x'] + beta['v_x'] #x2
        x_u[..., 3] = x_u[..., 3] * alpha['th_x'] + beta['th_x'] #x3
        x_u[..., 4] = x_u[..., 4] * alpha['o_x'] + beta['o_x'] #x4

        x_u[..., 5] = x_u[..., 5] * alpha['y'] + beta['y'] #x5
        x_u[..., 6] = x_u[..., 6] * alpha['v_y'] + beta['v_y'] #x6
        x_u[..., 7] = x_u[..., 7] * alpha['th_y'] + beta['th_y'] #x7
        x_u[..., 8] = x_u[..., 8] * alpha['o_y'] + beta['o_y'] #x8

        x_u[..., 9] = x_u[..., 9] * alpha['z'] + beta['z'] #x9
        x_u[..., 10] = x_u[..., 10] * alpha['v_z'] + beta['v_z'] #x10

        x_u[..., 11] = x_u[..., 11] * alpha['pu_x'] + beta['pu_x'] #x11
        x_u[..., 12] = x_u[..., 12] * alpha['pu_y'] + beta['pu_y'] #x12
        x_u[..., 13] = x_u[..., 13] * alpha['pu_z'] + beta['pu_z'] #x13
        
        n0 = 10     # Angular dynamics parameters
        d1 = 8
        d0 = 10
    
        kT = 0.91   # Thrust coefficient (vertical direction)
        g = 9.81    # Acceleration due to gravity (for convenience)
        m = 1.3     # Mass
        
        #Controls
        u1 = math.pi / 9
        u2 = math.pi / 9
        u3 = 1.5*g

        # Hamiltonian
        ham = dudx[...,0]*x_u[...,2] + dudx[...,1]*(g*torch.tan(x_u[...,3])) + dudx[...,2]*(-d1*x_u[...,3] + x_u[...,4]) + dudx[...,3]* (-d0*x_u[...,3]) #x
        ham = ham + dudx[...,4]*x_u[...,6] + dudx[...,5]*(g*torch.tan(x_u[...,7])) + dudx[...,6]*(-d1*x_u[...,7] + x_u[...,8]) + dudx[...,7]* (-d0*x_u[...,7]) #y
        ham = ham + dudx[..., 8]*x_u[...,10] + dudx[...,9]*(-g)
        ham = ham + (x_u[..., 11]*torch.abs(dudx[...,0])) + (x_u[..., 12]*torch.abs(dudx[...,4])) + (x_u[..., 13]*torch.abs(dudx[...,8]))
        ham = ham - n0*u1*torch.abs(dudx[..., 3]) - n0*u2*torch.abs(dudx[..., 7]) - kT*u3*torch.abs(dudx[..., 9])
        return ham
    
    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions
        angle_index = [3,7] # Index of the angle state

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Sample some points to impose the boundary coditions
        if self.periodic_boundary:
            # import ipdb; ipdb.set_trace()
            coords_angle = torch.zeros(self.N_boundary_pts, 1).uniform_(math.pi-0.001, self.alpha['th_x'] + self.beta['th_x']) # Sample near the right boundary
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated - self.beta['th_x'])/self.alpha['th_x']
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, 3] = coords_angle_concatenated_normalized[..., 0]
            coords[:2*self.N_boundary_pts, 7] = coords_angle_concatenated_normalized[..., 0]

        # Add some samples that are inside the target set
        #if self.sample_inside_target:
        #    target_coords = coords[:self.N_inside_target_samples] * 1.0 
        #    target_coords[..., 1:6] = self.sample_inside_target_set()
        #    #target_coords[-self.N_inside_target_samples//6:, 0] = start_time # Ensuring that some of the target set samples are the initial time
        #    coords = torch.cat((coords, target_coords), dim=0)

        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_IC(coords_var[..., 1:])
            
            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))

            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:11]
        else:
            boundary_values = self.compute_IC(coords[..., 1:])

            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.diffModel:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx_grads': lx_grads}
        else:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
        
class Q10DXY(Dataset):
    #     Constructor for a 10D quadrotor
    #  
    #       Dynamics of the 10D Quadrotor
    #           \dot x_1 = x_2 - d_1
    #           \dot x_2 = g * tan(x_3)
    #           \dot x_3 = -d1 * x_3 + x_4
    #           \dot x_4 = -d0 * x_3 + n0 * u1
    #           \dot x_5 = x_6 - d_2
    #           \dot x_6 = g * tan(x_7)
    #           \dot x_7 = -d1 * x_7 + x_8
    #           \dot x_8 = -d0 * x_7 + n0 * u2
    #           \dot x_9 = x_10 - d_3
    #           \dot x_10 = kT * u3
    #                uMin <= [u1; u2; u3] <= uMax
    #                dMin <= [d1; d2; d3] <= dMax
    def __init__(self, numpoints,
                 tMin=0.0, tMax=5, 
        pretrain=False, counter_start=0, counter_end=100e3, pretrain_iters=2000, 
        num_src_samples=1000, periodic_boundary=True, diffModel=False):
        super().__init__()
        torch.manual_seed(0)
        self.pretrain = pretrain
        self.periodic_boundary = periodic_boundary
        self.diffModel = diffModel
        #self.sample_inside_target = True
        self.numpoints = numpoints

        #Dynamics parameters
        self.num_states = 5 #(4D + P1D)

        #Time parameters
        self.tMax = tMax
        self.tMin = tMin


        # Normalization for states and time
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x, v_x, theta_x, omega_x, y, v_y, theta_y, omega_y, z, v_z]
        self.alpha = {}
        self.beta = {}

        self.alpha['x'] = 5.0 # x in [-5, 5] range
        self.alpha['v_x'] = 5.0 # v_x in [0, 10] range
        self.alpha['th_x'] = 1.0*math.pi #[-pi, pi] range
        self.alpha['o_x'] =  5.0 #o_y in [0, 10] range

        self.alpha['time'] = 1 #time is in [0, 1] range

        self.beta['x'] = 0
        self.beta['v_x'] = 5.0
        self.beta['th_x'] = 0
        self.beta['o_x'] =  5.0

        self.alpha['pu_x'] = 2.5 #planner control is in [0, 5] range
        self.beta['pu_x'] = 2.5

        # Normalization for the value function
        self.norm_to = 0.02 #Ultimate range after normalization [-0.02, 0.02]
        self.mean = 12.5  #Max val of l(x), and min val of l(x), find mean & var
        self.var = 12.5 # 
        #( ( v - mean ) / var  ) * norm_to
        #Pick mean and var so that this output is between [-1, 1]

        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2
        self.N_inside_target_samples = num_src_samples*5

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.zeroTensor = torch.tensor(0.).cuda()

    def __len__(self):
        return 1
    
    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[..., 0] = state_coords_unnormalized[..., 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[..., 1] = state_coords_unnormalized[..., 1] * self.alpha['v_x'] + self.beta['v_x']
        state_coords_unnormalized[..., 2] = state_coords_unnormalized[..., 2] * self.alpha['th_x'] + self.beta['th_x']
        state_coords_unnormalized[..., 3] = state_coords_unnormalized[..., 3] * self.alpha['o_x'] + self.beta['o_x']


        # positional states from start
        #boundary_values = torch.norm(state_coords_unnormalized[..., 0], dim=-1, keepdim=True) #One Norm
        boundary_values = state_coords_unnormalized[..., 0]**2 #Quad Cost
        boundary_values = boundary_values.unsqueeze(dim=-1)   # Add a new dimension
        return boundary_values

    def compute_overall_ham(self, x, dudx):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x'] #p1
        dudx[..., 1] = dudx[..., 1] / alpha['v_x'] #p2
        dudx[..., 2] = dudx[..., 2] / alpha['th_x'] #p3
        dudx[..., 3] = dudx[..., 3] / alpha['o_x'] #p4
    
        # Scale the states appropriately.
        x_u = x * 1.0
        x_u[..., 1] = x_u[..., 1] * alpha['x'] + beta['x'] #x1
        x_u[..., 2] = x_u[..., 2] * alpha['v_x'] + beta['v_x'] #x2
        x_u[..., 3] = x_u[..., 3] * alpha['th_x'] + beta['th_x'] #x3
        x_u[..., 4] = x_u[..., 4] * alpha['o_x'] + beta['o_x'] #x4

        #parameter
        x_u[..., 5] = x_u[..., 5] * alpha['pu_x'] + beta['pu_x'] #x5
        
        n0 = 10     # Angular dynamics parameters
        d1 = 8
        d0 = 10
    
        kT = 0.91   # Thrust coefficient (vertical direction)
        g = 9.81    # Acceleration due to gravity (for convenience)
        m = 1.3     # Mass
        
        #Controls
        u = math.pi / 9

        # Hamiltonian
        ham = dudx[...,0]*x_u[...,2] + dudx[...,1]*(g*torch.tan(x_u[...,3])) + dudx[...,2]*(-d1*x_u[...,3] + x_u[...,4]) + dudx[...,3]* (-d0*x_u[...,3]) #x
        ham = ham + (x_u[..., 5]*torch.abs(dudx[...,0])) #Planner Velocity
        ham = ham - n0*u*torch.abs(dudx[..., 3])
        return ham
    
    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions
        angle_index = 3 # Index of the angle state

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Sample some points to impose the boundary coditions
        if self.periodic_boundary:
            # import ipdb; ipdb.set_trace()
            coords_angle = torch.zeros(self.N_boundary_pts, 1).uniform_(math.pi-0.001, self.alpha['th_x'] + self.beta['th_x']) # Sample near the right boundary
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated - self.beta['th_x'])/self.alpha['th_x']
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, angle_index] = coords_angle_concatenated_normalized[..., 0]

        # Add some samples that are inside the target set
        #if self.sample_inside_target:
        #    target_coords = coords[:self.N_inside_target_samples] * 1.0 
        #    target_coords[..., 1:6] = self.sample_inside_target_set()
        #    #target_coords[-self.N_inside_target_samples//6:, 0] = start_time # Ensuring that some of the target set samples are the initial time
        #    coords = torch.cat((coords, target_coords), dim=0)

        # Compute the initial value function
        if self.diffModel:
            coords_var = coords.clone().detach().requires_grad_(True)
            boundary_values = self.compute_IC(coords_var[..., 1:])
            
            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))

            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:5]
        else:
            boundary_values = self.compute_IC(coords[..., 1:])

            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.diffModel:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx_grads': lx_grads}
        else:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
        
class Q10DZ(Dataset):
    #     Constructor for a 10D quadrotor
    #  
    #       Dynamics of the 10D Quadrotor
    #           \dot x_1 = x_2 - d_1
    #           \dot x_2 = g * tan(x_3)
    #           \dot x_3 = -d1 * x_3 + x_4
    #           \dot x_4 = -d0 * x_3 + n0 * u1
    #           \dot x_5 = x_6 - d_2
    #           \dot x_6 = g * tan(x_7)
    #           \dot x_7 = -d1 * x_7 + x_8
    #           \dot x_8 = -d0 * x_7 + n0 * u2
    #           \dot x_9 = x_10 - d_3
    #           \dot x_10 = kT * u3
    #                uMin <= [u1; u2; u3] <= uMax
    #                dMin <= [d1; d2; d3] <= dMax
    def __init__(self, numpoints,
                 tMin=0.0, tMax=5, 
        pretrain=False, counter_start=0, counter_end=100e3, pretrain_iters=2000, 
        num_src_samples=1000, periodic_boundary=False, diffModel=False):
        super().__init__()
        torch.manual_seed(0)
        self.pretrain = pretrain
        self.periodic_boundary = periodic_boundary
        self.diffModel = diffModel
        #self.sample_inside_target = True
        self.numpoints = numpoints

        #Dynamics parameters
        self.num_states = 3

        #Time parameters
        self.tMax = tMax
        self.tMin = tMin


        # Normalization for states and time
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x, v_x, theta_x, omega_x, y, v_y, theta_y, omega_y, z, v_z]
        self.alpha = {}
        self.beta = {}

        self.alpha['z'] = 5.0 # z in [-5, 5] range
        self.alpha['v_z'] = 5.0 # v_z in [0, 10] range

        self.beta['z'] = 0
        self.beta['v_z'] = 5.0

        self.alpha['pu_z'] = 2.5
        self.beta['pu_z'] = 2.5
        # Normalization for the value function
        self.norm_to = 0.02 #Ultimate range after normalization [-0.02, 0.02]
        self.mean = 12.50  #Max val of l(x), and min val of l(x), find mean & var
        self.var = 12.50 # 
        #( ( v - mean ) / var  ) * norm_to
        #Pick mean and var so that this output is between [-1, 1]

        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2
        self.N_inside_target_samples = num_src_samples*5

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.zeroTensor = torch.tensor(0.).cuda()

    def __len__(self):
        return 1
    
    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[..., 0] = state_coords_unnormalized[..., 0] * self.alpha['z'] + self.beta['z']
        state_coords_unnormalized[..., 1] = state_coords_unnormalized[..., 1] * self.alpha['v_z'] + self.beta['v_z']

        # positional states from start
        boundary_values = state_coords_unnormalized[..., 0]**2 #Quad Cost
        boundary_values = boundary_values.unsqueeze(dim=-1)   # Add a new dimension
        return boundary_values
    
    def compute_overall_ham(self, x, dudx):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['z'] #p1
        dudx[..., 1] = dudx[..., 1] / alpha['v_z'] #p2
    
        # Scale the states appropriately.
        x_u = x * 1.0
        x_u[..., 1] = x_u[..., 1] * alpha['z'] + beta['z'] #x1
        x_u[..., 2] = x_u[..., 2] * alpha['v_z'] + beta['v_z'] #x2

        x_u[..., 3] = x_u[..., 3] * alpha['pu_z'] + beta['pu_z'] #x3
        
        n0 = 10     # Angular dynamics parameters
        d1 = 8
        d0 = 10
    
        kT = 0.91   # Thrust coefficient (vertical direction)
        g = 9.81    # Acceleration due to gravity (for convenience)
        m = 1.3     # Mass
        
        #Controls
        u3 = 1.5*g

        # Hamiltonian
        ham = dudx[..., 0]*x_u[...,2] + dudx[...,1]*(-g)
        ham = ham + (x_u[..., 3]*torch.abs(dudx[...,0]))
        ham = ham - kT*u3*torch.abs(dudx[..., 1])
        return ham
    
    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions
        angle_index = [] # Index of the angle state

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Sample some points to impose the boundary coditions
        if self.periodic_boundary:
            # import ipdb; ipdb.set_trace()
            coords_angle = torch.zeros(self.N_boundary_pts, 1).uniform_(math.pi-0.001, self.alpha['th_x'] + self.beta['th_x']) # Sample near the right boundary
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated - self.beta['th_x'])/self.alpha['th_x']
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, 3] = coords_angle_concatenated_normalized[..., 0]
            coords[:2*self.N_boundary_pts, 7] = coords_angle_concatenated_normalized[..., 0]

        # Add some samples that are inside the target set
        #if self.sample_inside_target:
        #    target_coords = coords[:self.N_inside_target_samples] * 1.0 
        #    target_coords[..., 1:6] = self.sample_inside_target_set()
        #    #target_coords[-self.N_inside_target_samples//6:, 0] = start_time # Ensuring that some of the target set samples are the initial time
        #    coords = torch.cat((coords, target_coords), dim=0)

        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_IC(coords_var[..., 1:])
            
            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))

            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:3]
        else:
            boundary_values = self.compute_IC(coords[..., 1:])

            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.diffModel:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx_grads': lx_grads}
        else:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}



