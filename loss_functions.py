import torch
import torch.nn.functional as F

import diff_operators
import modules, utils

import math
import numpy as np

def initialize_hji_Q10DP3D(dataset, minWith, diffModel_mode):
    # Normalization parameters
    alpha = dataset.alpha
    beta = dataset.beta

    # Ham function
    compute_overall_ham = dataset.compute_overall_ham

    # Other flags
    periodic_boundary = dataset.periodic_boundary
    num_boundary_pts = dataset.N_boundary_pts
    normalized_zero_value = -dataset.mean * dataset.norm_to/dataset.var #The normalized value corresponding to V=0

    diffModel = dataset.diffModel
    def hji_Q10DP3D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]        

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # Compute the spatial and time derivatives of the value function
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:14]

            # Adjust the spatial gradient of lx if needed
            if diffModel:
                dudx = dudx + gt['lx_grads']
                if diffModel_mode == 'mode1':
                    diff_from_lx = y - normalized_zero_value
                elif diffModel_mode == 'mode2':
                    diff_from_lx = y
                else:
                    raise NotImplementedError   
            else:
                diff_from_lx = y - source_boundary_values

            # Compute the Hamiltonian
            ham = compute_overall_ham(x, dudx)

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, min=0.0)
            if minWith == 'maxVwithV0': #max between Hamiltonian and cost function
                 ham = torch.clamp(ham, min=0.0)
            diff_constraint_hom = dudt - ham
            if minWith == 'target': #hamiltonian post processor
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], diff_from_lx)

        # Boundary loss
        if diffModel:
            dirichlet = y[dirichlet_mask]
        else:
            dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        if periodic_boundary:
            # import ipdb; ipdb.set_trace()
            periodic_boundary_loss = y[:, :num_boundary_pts] - y[:, num_boundary_pts:2*num_boundary_pts]
            return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 75e2,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                    'periodicity': torch.abs(periodic_boundary_loss).sum() * batch_size * 50 / 75e2}
        else:
            return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 75e2,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_Q10DP3D

def initialize_hji_Q10DXY(dataset, minWith, diffModel_mode):
    # Normalization parameters
    alpha = dataset.alpha
    beta = dataset.beta

    # Ham function
    compute_overall_ham = dataset.compute_overall_ham

    # Other flags
    periodic_boundary = dataset.periodic_boundary
    num_boundary_pts = dataset.N_boundary_pts
    diffModel = dataset.diffModel
    def hji_Q10DXY(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]        

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # Compute the spatial and time derivatives of the value function
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:5]

            # Adjust the spatial gradient of lx if needed
            if diffModel:
                dudx = dudx + gt['lx_grads']
                if diffModel_mode == 'mode1':
                    diff_from_lx = y #what
                elif diffModel_mode == 'mode2':
                    diff_from_lx = y
                else:
                    raise NotImplementedError   
            else:
                diff_from_lx = y - source_boundary_values

            # Compute the Hamiltonian
            ham = compute_overall_ham(x, dudx) 

            # Scale the time derivative appropriately
            #ham = ham * alpha['time']

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)
            if minWith == 'maxVwithV0':
                 ham = torch.clamp(ham, max=0.0)
            diff_constraint_hom = dudt - ham 
            #diff_constraint_hom = torch.min(diff_constraint_hom[:, :, None], diff_from_lx)
            if minWith == 'target': #hamiltonian post processor
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], diff_from_lx)

        # Boundary loss
        if diffModel:
            dirichlet = y[dirichlet_mask]
        else:
            dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        if periodic_boundary:
            # import ipdb; ipdb.set_trace()
            periodic_boundary_loss = y[:, :num_boundary_pts] - y[:, num_boundary_pts:2*num_boundary_pts]
            return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 75e2,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                    'periodicity': torch.abs(periodic_boundary_loss).sum() * batch_size * 50 / 75e2}
        else:
            return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 75e2,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_Q10DXY

def initialize_hji_Q10DZ(dataset, minWith, diffModel_mode):
    # Normalization parameters
    alpha = dataset.alpha
    beta = dataset.beta

    # Ham function
    compute_overall_ham = dataset.compute_overall_ham

    # Other flags
    periodic_boundary = dataset.periodic_boundary
    num_boundary_pts = dataset.N_boundary_pts
    diffModel = dataset.diffModel
    def hji_Q10DZ(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]        

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # Compute the spatial and time derivatives of the value function
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:3]

            # Adjust the spatial gradient of lx if needed
            if diffModel:
                dudx = dudx + gt['lx_grads']
                if diffModel_mode == 'mode1':
                    diff_from_lx = y #what
                elif diffModel_mode == 'mode2':
                    diff_from_lx = y
                else:
                    raise NotImplementedError   
            else:
                diff_from_lx = y - source_boundary_values

            # Compute the Hamiltonian
            ham = compute_overall_ham(x, dudx) 

            # Scale the time derivative appropriately
            #ham = ham * alpha['time']

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, min=0.0)
            if minWith == 'maxVwithV0': #max between Hamiltonian and cost function
                 ham = torch.clamp(ham, min=0.0)
            diff_constraint_hom = dudt - ham
            if minWith == 'target': #hamiltonian post processor
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], diff_from_lx)

        # Boundary loss
        if diffModel:
            dirichlet = y[dirichlet_mask]
        else:
            dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        if periodic_boundary:
            # import ipdb; ipdb.set_trace()
            periodic_boundary_loss = y[:, :num_boundary_pts] - y[:, num_boundary_pts:2*num_boundary_pts]
            return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 75e2,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                    'periodicity': torch.abs(periodic_boundary_loss).sum() * batch_size * 50 / 75e2}
        else:
            return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 75e2,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_Q10DZ