import torch

torch.cuda.is_available = lambda: False

from pprint import pprint
from typing import Callable
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import RnnModel
import Pipeline
import os
import scipy.io as scio
import functions
from datetime import datetime
# from Intergration import Deadreckoning, Error_state, Error_state_8, Error_state_8_LS
# %matplotlib widget

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Choose GPU core
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(3)  # make experiment repeatable
# =============================================================================
#                         Check GPU avaliability
# =============================================================================
if torch.cuda.is_available():
    dev = torch.device("cuda")  # cuda:1 cuda:2....etc.
    torch.set_default_dtype(torch.float64)
    datatype = np.float64
    torch.set_printoptions(precision=16)
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    datatype = np.float64
    torch.set_printoptions(precision=16)
    print("Running on the CPU")


today_date = datetime.today().strftime("%m.%d.%y")[0:5].replace('.','')
currt_folder = os.getcwd()
# =============================================================================
#                      Step1 Load data and set parameter
# =============================================================================

#realdata_3d_10hz
#simudata_bias_3d_10hz
#simudata_MEMS_3d_10hz
#simudata_bias_White_3d_10hz
#GIVE_real_data
# GIVE_real_data_0316
#GIVE_real_data_0316_60s
# simudata_Errorfree_3d_10hz

dataset_type = 'simudata_MEMS_3d_10hz.mat'
indx_MB = True

# 
# operation = 'TRAIN'
 
operation = 'TEST'
in_model = '0823_MEMS_withgrad_shuffle_lstm111_1000_pvab_lkrelu_alltorchnorm_m2m_noifimuatt01001_loss123'
Longer_test_traj_indx = True
%matplotlib widget

# operation = 'TRAIN_USEOLD'
# in_model = '0817_bias_withgrad_shuffle_lstm111_200_pvab_lkrelu_alltorchnorm_m2m_ifimuatt01001_loss23_1'
# out_model = '0817_bias_withgrad_shuffle_lstm111_200_pvab_lkrelu_alltorchnorm_m2m_ifimuatt01001_loss23_2'


nograd_sd = False # True: with torch.no_grad, nograd    False: none, withgrad
idx_shuffle = True
indx_add_measerr = False
indx_train_gnssgap = True

idx_lossweight_coeff = ['1','1','1']

idx_feedback_type = 'pvab'
# idx_feedback_type = 'pvb'
# idx_other_settings = '_feedavpb_lkrelu_alltorchnorm_m2m_noifimuatt0101'
idx_other_settings = '_lkrelu_alltorchnorm_m2m_ifimuatt01001_loss23'


idx_train_batch_size = 20
idx_cut_test_traj = 1

input_dim = 9 + 9 + 6 + 6
hidden_dim =256
n_layers = 4
linearfc2_dim = 512
linearfc3_dim = 256
output_dim = 54
droupout_rate = 0
recurrent_kind = 'lstm'  # 'rnn' 'gru' 'lstm'


idx_num_epochs = 200
idx_learning_rate =1e-5
idx_weight_decay = 1e-8
scheduler = "cosine_annealing 200"
# scheduler = "step 100 0.1"

# scheduler = "None"


# =============================================================================

data_path = currt_folder + '/data/' + dataset_type


if operation == 'TRAIN':

    if 'bias' in data_path:
        idx_dataset_type = '_bias'
    elif 'MEMS' in data_path:
        idx_dataset_type = '_MEMS'
    elif 'real' in data_path:
        idx_dataset_type = '_real'
    elif 'bias_White' in data_path:
        idx_dataset_type = '_bias_White'
    elif 'Errorfree' in data_path:
        idx_dataset_type = '_Errorfree'
    elif '60s' in data_path:
        idx_dataset_type = '_real_60s'
    

    if nograd_sd == False:
        idx_dr_grad = '_withgrad_'
    else:
        idx_dr_grad = '_nograd_'
    if idx_shuffle == False:
        idx_shuffle_str = '_'
    else:
        idx_shuffle_str = 'shuffle_'

    if indx_add_measerr == True:
        idx_add_err = 'add_measerr'
    else:
        idx_add_err = ''

    if indx_train_gnssgap:
        inx_train_gnssgap = 'gnssgap_'
    else:
        inx_train_gnssgap = ''

    idx_lossweight_coeff_str = ''.join(idx_lossweight_coeff)    
    out_model = today_date + idx_dataset_type + idx_dr_grad + idx_shuffle_str + inx_train_gnssgap + idx_add_err + recurrent_kind + idx_lossweight_coeff_str + '_' + str(idx_num_epochs) + '_' + idx_feedback_type + idx_other_settings
    out_model_path = currt_folder + '/model/' + out_model + '.pt'

    LoadModel = False
    TrainModel = True
    SaveModel = True
    print(out_model)

elif operation == 'TEST':

    in_model_path = currt_folder + '/model/' + in_model + '.pt'
    print(in_model)

    LoadModel = True
    TrainModel = False
    SaveModel = False

elif operation == 'TRAIN_USEOLD':
    in_model_path = currt_folder + '/model/' + in_model + '.pt'
    print(in_model)
    LoadModel = True
    TrainModel = True
    SaveModel = True
    out_model_path = currt_folder + '/model/' + out_model + '.pt'

# =============================================================================

data = scio.loadmat(data_path)
dataset = data["dataset"]


pos_meas =  torch.from_numpy(dataset[0, 0]["pos_meas_ecef"].astype(datatype))
vel_meas =  torch.from_numpy(dataset[0, 0]["vel_meas_ecef"].astype(datatype))

if indx_add_measerr:
    if 'simu' in data_path:
        manual_error_pos = torch.rand((pos_meas.shape[0],pos_meas.shape[1],pos_meas.shape[2])) * 2
        manual_error_vel = torch.rand((pos_meas.shape[0],pos_meas.shape[1],pos_meas.shape[2]))
        pos_meas = pos_meas + manual_error_pos
        vel_meas = vel_meas + manual_error_vel

# data = scio.loadmat("data/8020_8shape_10hz_100s_bias_randw_mis.mat")
# # data = scio.loadmat("data/8020_8shape_10hz_100s_bias_randw_mis.mat")

# dataset = data["dataset"]

Fs = int(dataset[0, 0]["Fs"])  # IMU freq
Fs_meas = int(dataset[0, 0]["Fs_range"])  # range meas freq
tarj_length = int(dataset[0, 0]["tarj_length"])
train_num = int(dataset[0, 0]["train_num"]) + 1
test_num = int(dataset[0, 0]["test_num"])

time_traj = torch.from_numpy(dataset[0, 0]["time_traj_real"]).float()
position_ecef = torch.from_numpy(dataset[0, 0]["position_ecef"].astype(datatype))
velocity_ecef = torch.from_numpy(dataset[0, 0]["velocity_ecef"].astype(datatype))
position_ned = torch.from_numpy(dataset[0, 0]["position_ned"].astype(datatype))
velocity_ned = torch.from_numpy(dataset[0, 0]["velocity_ned"].astype(datatype))

position_llh = torch.from_numpy(dataset[0, 0]["position_llh"].astype(datatype))




acceleration = torch.from_numpy(dataset[0, 0]["acceleration"].astype(datatype))
# orientation_euler = torch.from_numpy(dataset[0, 0]["orientation_euler"]).float()
orientation_euler_rad = torch.from_numpy(dataset[0, 0]["orientation_euler_rad"].astype(datatype))

accbody = torch.from_numpy(dataset[0, 0]["acc_meas"].astype(datatype))
angularVelocity = torch.from_numpy(dataset[0, 0]["angularv_meas"].astype(datatype))

att_ecef_euler_rad = torch.zeros(orientation_euler_rad.shape[0],orientation_euler_rad.shape[1],3)
reshape_C_b_e = torch.zeros(orientation_euler_rad.shape[0],orientation_euler_rad.shape[1],9)
for ii in range(orientation_euler_rad.shape[0]):
    for jj in range(orientation_euler_rad.shape[1]):
        est_C_b_n = functions.euler_to_CTM(orientation_euler_rad[ii,jj]).T
        _,_,est_C_b_e= functions.geo_ned2ecef(position_llh[ii,jj], velocity_ned[ii,jj], est_C_b_n)
        att_ecef_euler_rad[ii,jj] = functions.CTM_to_euler(est_C_b_e.T)
        reshape_C_b_e[ii,jj] = est_C_b_e.reshape(9)
# time_meas = torch.from_numpy(dataset[0, 0]["time_meas"]).float()
# rangepos1 = torch.from_numpy(dataset[0, 0]['rangepos1'].astype(datatype))
# rangepos2 = torch.from_numpy(dataset[0, 0]['rangepos2'].astype(datatype))
# rangepos3 = torch.from_numpy(dataset[0, 0]['rangepos3'].astype(datatype))
# allref = torch.cat((rangepos1, rangepos2, rangepos3), dim=1).to(dev)
# rangeacc = float(dataset[0, 0]["rangeacc"])
# r1 = torch.unsqueeze((torch.from_numpy(dataset[0, 0]["r1"].astype(datatype))), dim=-1)
# r2 = torch.unsqueeze((torch.from_numpy(dataset[0, 0]["r2"].astype(datatype))), dim=-1)
# r3 = torch.unsqueeze((torch.from_numpy(dataset[0, 0]["r3"].astype(datatype))), dim=-1)

# b_a = torch.from_numpy(dataset[0, 0]['b_a'].astype(datatype))[0][0]
# b_g = torch.from_numpy(dataset[0, 0]['b_g'].astype(datatype))[0][0]

# pos_meas =  torch.from_numpy(dataset[0, 0]['pos_meas'].astype(datatype))

# =============================================================================
#                   Step2 Using PyTorch Dataset and DataLoader
# =============================================================================
# Linear Dataset
# X = torch.cat((r1, r2, r3), dim=2).to(dev)
X = torch.cat((pos_meas, vel_meas), dim=2).to(dev)
T = torch.cat((position_ecef, velocity_ecef, position_ned, velocity_ned, position_llh, acceleration, orientation_euler_rad, att_ecef_euler_rad, reshape_C_b_e), dim=2).to(dev)
IMU = torch.cat((accbody, angularVelocity), dim=2).to(dev)
IMU = IMU.reshape(IMU.shape[0], -1, Fs, 6)
# Circular Dataset
# X = torch.from_numpy(np.load("data/rangemeas_circle.npy")).to(dev)
# T = torch.from_numpy(np.load("data/trajectory_ref_circle.npy")).to(dev)

# Mix Dataset
# X = torch.from_numpy(np.load("data/rangemeas_Mix_4d.npy")).to(dev)
# T = torch.from_numpy(np.load("data/trajectory_ref_Mix_4d.npy")).to(dev)
# train_num=20
''' the train batch could be larger'''
train_batch_size = idx_train_batch_size  # choose batch size
train_val_splitter = 0.9

train_features = X[0 : int(train_num * train_val_splitter)].to(dev)
train_IMU = IMU[0 : int(train_num * train_val_splitter)].to(dev)
train_targets = T[0 : int(train_num * train_val_splitter)].to(dev)
# train_targets = T2[0:trainsets_num]

val_features = X[int(train_num * train_val_splitter) : train_num].to(dev)
val_IMU = IMU[int(train_num * train_val_splitter) : train_num].to(dev)
val_targets = T[int(train_num * train_val_splitter) : train_num].to(dev)

# val_features = train_features
# val_targets = train_targets

# 20s traj to test
test_time_traj = time_traj[train_num:].to(dev)
test_features = X[train_num:].to(dev)
test_IMU = IMU[train_num:].to(dev)
test_targets = T[train_num:].to(dev)
# test_targets = T2[trainsets_num:]

if Longer_test_traj_indx:
    # Longer test traj
    # indx_test = idx_cut_test_traj
    # # test_time_traj = test_time_traj.reshape(indx_test,-1)
    # test_features = train_features.reshape(indx_test,-1,6)
    # test_IMU = train_IMU.reshape(indx_test,-1,Fs,6)
    # test_targets = train_targets.reshape(indx_test,-1,33)


    # # Longer test traj
    indx_test = idx_cut_test_traj
    # test_time_traj = test_time_traj.reshape(indx_test,-1)
    test_features = test_features.reshape(indx_test,-1,6)
    test_IMU = test_IMU.reshape(indx_test,-1,Fs,6)
    test_targets = test_targets.reshape(indx_test,-1,33)

print("data loaded")


class Range_INS_Dataset(Dataset):
    def __init__(self, features, targets, imu_meas):
        self.features = features
        self.targets = targets  
        self.imu_meas = imu_meas
        self.num_traj = features.shape[0]
        self.time_step = features.shape[1]

    def __getitem__(self, index):
        return self.features[index], self.targets[index], self.imu_meas[index]

    def __len__(self):
        return self.num_traj


train_dataset = Range_INS_Dataset(train_features, train_targets, train_IMU)

if idx_shuffle == True:
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True
    )
else:
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=False
    )  

'''Here, if shuffle = false, we only take the fisrt train_batch_size trajectory to train'''

val_dataset = Range_INS_Dataset(val_features, val_targets, val_IMU)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

test_dataset = Range_INS_Dataset(test_features, test_targets, test_IMU)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# =============================================================================
#                               Init Net
# =============================================================================
"""est_P"""

# m = 9
# # n = 6

# input_dim = m
# linearfc1_dim = m * 50
# hidden_dim = (m**2) * 10 * 1
# n_layers = 2
# linearfc2_dim = m * 20
# output_dim = m * m

# net = Rnn_Kalman_Model.LC_est_P(
#     input_dim, linearfc1_dim, hidden_dim, n_layers, linearfc2_dim, output_dim
# )
# # not really necessary since default tensor type is already set appropriately
# net.to(dev)
# net = torch.load("test_P.pt", map_location=dev)
"""est_P_Q_R"""

# m = 9
# n = 6

# input_dim_1 = m 
# linearfc1_dim_1 = m * 100
# hidden_dim_1 = (m ** 2) * 10
# n_layers_1 = 16
# linearfc2_dim_1 = m * 20
# output_dim_1 = m * m

# input_dim_2 = m
# linearfc1_dim_2 = m * 100
# hidden_dim_2 = (m ** 2) * 10
# n_layers_2 = 1
# linearfc2_dim_2 = m * 20
# output_dim_2 = m * m

# input_dim_3 = n
# linearfc1_dim_3 = n * 50
# hidden_dim_3 = (n ** 2) * 10
# n_layers_3 = 1
# linearfc2_dim_3 = n * 20
# output_dim_3 = n * n

# net = Rnn_Kalman_Model.LC_est_P_Q_R(input_dim_1, linearfc1_dim_1, hidden_dim_1, n_layers_1, linearfc2_dim_1, output_dim_1,
#                                     input_dim_2, linearfc1_dim_2, hidden_dim_2, n_layers_2, linearfc2_dim_2, output_dim_2,
#                                     input_dim_3, linearfc1_dim_3, hidden_dim_3, n_layers_3, linearfc2_dim_3, output_dim_3)

# net = torch.load('test_Q_P_R.pt', map_location=dev)
"""est_KG"""

net = RnnModel.LC_est_KG(
    input_dim,
    hidden_dim,
    n_layers,
    linearfc2_dim,
    linearfc3_dim,
    output_dim,
    droupout_rate,
    recurrent_kind,
    Fs,
    idx_feedback_type,
)
net.to(dev)

if LoadModel:
    net.load_state_dict(torch.load(in_model_path))

# =============================================================================
#                           Step4 Train the model
# =============================================================================

def make_scheduler(scheduler):
    if scheduler == "None":
        return None
    scheduler_kind, *scheduler_params = scheduler.split(" ")
    scheduler_params = [float(x) for x in scheduler_params]
    gen = None
    if scheduler_kind == "cosine_annealing":
        gen = CosineAnnealingLR
    elif scheduler_kind == "cosine_annealing_warm":
        gen = CosineAnnealingWarmRestarts
    elif scheduler_kind == "step":
        gen = StepLR
    else:
        raise ValueError("Invalid scheduler")
    return lambda optimizer: gen(optimizer, *scheduler_params)


scheduler = "cosine_annealing 500"
# scheduler = "step 100 0.1"

# scheduler = "None"

Trainer = Pipeline.Pipeline_LC(
    net,
    num_epochs=idx_num_epochs,
    learning_rate=idx_learning_rate,
    weight_decay=idx_weight_decay,
    loss_fn=nn.MSELoss(reduction="mean"),
    scheduler_generator=make_scheduler(scheduler),
    nograd = nograd_sd,
    lossweight_coeff = idx_lossweight_coeff,
    train_gnssgap = indx_train_gnssgap,
)
# use imu_time_interval = 1 for fast training, but increase error in dr, imu frequency is 100Hz, use all of it will be super slow
# Change float32 to 64 is very imporant when use time interval smaller that 1s. Accuary of DR changes from 70 to 0.001!

if TrainModel:
    timer = functions.Timer()
    Trainer.train_lc(train_loader, train_dataset, val_loader, val_dataset, Fs, Fs_meas)
    print(f"{timer.stop():.2f} sec")

if SaveModel:
    torch.save(net.state_dict(), out_model_path)


# =============================================================================
#                               Test
# =============================================================================
# todo: change to a test dataset
# this is fine as long as we don't use the validation dataset (i.e. as a stopping criterion)
# Change output to checklist to check several params
est_traj_nn, ref_traj, bias_history, est_traj_nn_llh, ref_traj_llh= Trainer.test_lc(test_loader, test_dataset, Fs, Fs_meas)

# KGain = check_list_test[0].cpu().detach().numpy()
# P = check_list_test[1].cpu().detach().numpy()
# Q = check_list_test[2].cpu().detach().numpy()
# R = check_list_test[3].cpu().detach().numpy()
# est_IMU_bias_new_array = est_IMU_bias_new.cpu().detach().numpy()

# est_att_ned_euler_rad = torch.zeros(est_traj_nn.shape[0],est_traj_nn.shape[1],3)
# for ii in range(est_traj_nn.shape[0]):
#     for jj in range(est_traj_nn.shape[1]):
#         est_C_b_e = functions.euler_to_CTM(est_traj_nn[ii,jj,6:]).T
#         _,_,est_att_ned_euler_rad[ii,jj]= functions.ecef2geo_ned(est_traj_nn[ii,jj,:3], est_traj_nn[ii,jj,3:6], est_C_b_e)

# =============================================================================
#                              Position Accuracy
# =============================================================================

def rem2pi(tensor: torch.Tensor) -> torch.Tensor:
    return torch.remainder(tensor, 2 * np.pi)


def loss(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    return (torch.sum(torch.norm(predicted - reference, dim=1)) / reference.shape[1]).item()


def loss_total(start: int, end: int, predicted: torch.Tensor,
               transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> float:
    return loss(transform(predicted[:, :, start:end]), transform(ref_traj[:, :, start:end]))


losses = {
    "position": {
        "lstm": loss_total(0, 3, est_traj_nn),
    },
    "velocity": {
        "lstm": loss_total(3, 6, est_traj_nn),
    },
    # "attitude": {
    #     "lstm": loss_total(4, 5, est_traj_nn, rem2pi),
    # }
}

print("Losses:")
pprint(losses)


# =============================================================================
#                                     Model Based
# =============================================================================


if indx_MB:

    if 'bias' in data_path:
        MB_dataset_type = 'bias'
    elif 'MEMS' in data_path:
        MB_dataset_type = 'MEMS'
    elif 'real' in data_path:
        MB_dataset_type = 'real'
    elif 'bias_White' in data_path:
        MB_dataset_type = 'bias_White'


    est_traj_nn_MB, ref_traj_MB, bias_history_MB, est_traj_nn_llh_MB, ref_traj_llh_MB= Trainer.GnssInsLooseCoupling(MB_dataset_type, test_loader, test_dataset, Fs, Fs_meas)


    losses = {
    "position": {
        "lstm": loss_total(0, 3, est_traj_nn_MB),
    },
    "velocity": {
        "lstm": loss_total(3, 6, est_traj_nn_MB),
    },
    # "attitude": {
    #     "lstm": loss_total(4, 5, est_traj_nn, rem2pi),
        # }
    }

    print("MB Losses:")
    pprint(losses)


# =============================================================================
#                                     Plot
# =============================================================================

est_traj_nn = est_traj_nn.detach().cpu().numpy()
ref_traj = ref_traj.detach().cpu().numpy()
bias_history = bias_history.detach().cpu().numpy()
est_traj_nn_llh = est_traj_nn_llh.numpy()
ref_traj_llh = ref_traj_llh.numpy()


# Pos LSTM
plt.figure()
plt.grid()
plt.title("Net position")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj_llh[k, :, 1], ref_traj_llh[k, :, 0], "b", label="Reference" if k == 0 else None)
    plt.plot(est_traj_nn_llh_MB[k, :, 1], est_traj_nn_llh_MB[k, :, 0], "g", label="MB" if k == 0 else None)
    plt.plot(est_traj_nn_llh[k, :, 1], est_traj_nn_llh[k, :, 0], "r", label="Net" if k == 0 else None)
    
plt.legend()

# Pos LSTM
plt.figure()
plt.grid()
plt.title("Net position Height")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj_llh[k, :, 2], "b", label="Reference"  if k == 0 else None)
    plt.plot(est_traj_nn_llh_MB[k, :, 2], "g", label="Net"  if k == 0 else None)
    plt.plot(est_traj_nn_llh[k, :, 2], "r", label="Net"  if k == 0 else None)
plt.legend()


# Pos LSTM
# plt.figure()
# plt.grid()
# plt.title("Net position")
# for k in range(5):  # est_traj_nn.shape[0]):
#     plt.plot(ref_traj[k, :, 1], ref_traj[k, :, 0], "b", label="Reference")
#     plt.plot(est_traj_nn[k, :, 1], est_traj_nn[k, :, 0], "r", label="Net")
# plt.legend()


plt.figure()
plt.grid()
plt.title("Attitude")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj[k, :, 21], "b", label="Reference x"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 22], "b", label="Reference y"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 23], "b", label="Reference z"  if k == 0 else None)

plt.legend()


# def rem2pi_np(array):
#     return np.remainder(array, 2 * np.pi)

plt.figure()
plt.grid()
plt.title("Attitude")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(est_traj_nn[k, :, 6], "r", label="Net x"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 7], "r", label="Net y"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 8], "r", label="Net z"  if k == 0 else None)

plt.legend()




# est_att_ned_euler_rad = est_att_ned_euler_rad.numpy()

# plt.figure()
# plt.grid()
# plt.title("Attitude")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(ref_traj[k, :, 18], "b", label="Reference x")
#     plt.plot(ref_traj[k, :, 19], "b", label="Reference y")
#     plt.plot(ref_traj[k, :, 20], "b", label="Reference z")

# plt.legend()

# plt.figure()
# plt.grid()
# plt.title("Attitude")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(est_att_ned_euler_rad[k, :, 0], "r", label="Net x")
#     plt.plot(est_att_ned_euler_rad[k, :, 1], "r", label="Net y")
#     plt.plot(est_att_ned_euler_rad[k, :, 2], "r", label="Net z")

# plt.legend()



# Pos LSTM
# Vel LSTM
# plt.figure()
# plt.grid()
# plt.title("Net velocity")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(est_traj_nn_llh[k, ::10, 3], label=r"Net $v_n$")
#     plt.plot(est_traj_nn_llh[k, ::10, 4], label=r"Net $v_e$")
#     plt.plot(est_traj_nn_llh[k, ::10, 5], label=r"Net $v_d$")
#     plt.plot(ref_traj_llh[k, ::10, 3], label=r"Reference $v_n$")
#     plt.plot(ref_traj_llh[k, ::10, 4], label=r"Reference $v_e$")
#     plt.plot(ref_traj_llh[k, ::10, 5], label=r"Reference $v_d$")

# plt.legend()



# Bias history
plt.figure()
plt.grid()
plt.title("Acc")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history[k, :, 0], "b", label=r"$Acc x$" if k == 0 else None)
    plt.plot(bias_history[k, :, 1], "r", label=r"$Acc y$" if k == 0 else None)
    plt.plot(bias_history[k, :, 2], "g", label=r"$Acc z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()

# Bias history
plt.figure()
plt.grid()
plt.title("Gyro")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history[k, :, 3], "b", label=r"$Gyro x$" if k == 0 else None)
    plt.plot(bias_history[k, :, 4], "r", label=r"$Gyro y$" if k == 0 else None)
    plt.plot(bias_history[k, :, 5], "g", label=r"$Gyro z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()

# Bias history
plt.figure()
plt.grid()
plt.title("Acc MB")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history_MB[k, :, 0], "b", label=r"$Acc x$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 1], "r", label=r"$Acc y$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 2], "g", label=r"$Acc z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()

# Bias history
plt.figure()
plt.grid()
plt.title("Gyro MB")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history_MB[k, :, 3], "b", label=r"$Gyro x$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 4], "r", label=r"$Gyro y$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 5], "g", label=r"$Gyro z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()


plt.figure()
plt.grid()
plt.title("C_b_e")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj[k, :, 25], "b", label="ref"  if k == 0 else None)
    plt.plot(est_traj_nn_MB[k, :, 10], "g", label="MB"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 10], "r", label="Net"  if k == 0 else None)
plt.legend()


# scio.savemat(
#     "data/0316real.mat",
#     mdict={
#         "ref_traj": ref_traj,
#         "est_traj_nn": est_traj_nn,
#         "ref_traj_llh": ref_traj_llh,
#         "est_traj_nn_llh": est_traj_nn_llh,
#         # "time_traj" : test_time_traj,
#     },
# )



# Calculate Position Error
# pos_err_NN_test = np.zeros(est_pos_test_array.shape[0])
# pos_err_LC_test = np.zeros(lc_array_cut.shape[0])
# for i in range(est_pos_test_array.shape[0]):
#     pos_err_NN_test[i] = (
#         (ref_array_cut[i, 1] - est_pos_test_array[i, 0]) ** 2
#         + (ref_array_cut[i, 2] - est_pos_test_array[i, 1]) ** 2
#         + (ref_array_cut[i, 3] - est_pos_test_array[i, 2]) ** 2
#     ) ** (1 / 2)
#     pos_err_LC_test[i] = (
#         (ref_array_cut[i, 1] - lc_array_cut[i, 1]) ** 2
#         + (ref_array_cut[i, 2] - lc_array_cut[i, 2]) ** 2
#         + (ref_array_cut[i, 3] - lc_array_cut[i, 3]) ** 2
#     ) ** (1 / 2)

# # Calculate LLH and Vel NED of test result
# est_pos_test_array_llh = np.zeros((est_pos_test_array.shape[0], 3))
# est_vel_test_array_ned = np.zeros((est_pos_test_array.shape[0], 3))
# for i in range(est_pos_test_array.shape[0]):
#     pos_llh, vel_ned = rnm.ecef2geo_ned_array(
#         est_pos_test_array[i, 0:3], est_vel_test_array[i, 0:3]
#     )
#     est_pos_test_array_llh[i, 0:3] = pos_llh.reshape(3)
#     est_vel_test_array_ned[i, 0:3] = vel_ned.reshape(3)

# # Plot Postioning Error
# rnm.plt.figure()
# rnm.plot([pos_err_NN_test, pos_err_LC_test], legend=("NN", "LC"), title='Position Error')
# # Plot Traj 
# rnm.plt.figure()
# rnm.plot(
#     [est_pos_test_array_llh[:, 1], lc_array_cut[:, 8], ref_array_cut[0:, 8]],
#     [est_pos_test_array_llh[0:, 0], lc_array_cut[:, 7], ref_array_cut[0:, 7]],
#     legend=("NN", "LC", "Ref"), title='Trajectory'
# )
# # Plot Velocity NED
# rnm.plt.figure()
# rnm.plot(
#     [
#         est_vel_test_array_ned[:, 0],
#         est_vel_test_array_ned[:, 1],
#         est_vel_test_array_ned[:, 2],
#     ], title='Velocity NED'
# )
# # Plot Attitude 
# rnm.plt.figure()
# # rnm.plot([est_att_test_array[:,0],lc_array_cut[:,13],ref_array_cut[0:,13],
# #           est_att_test_array[:,1],lc_array_cut[:,14],ref_array_cut[0:,14],
# #           est_att_test_array[:,2],lc_array_cut[:,15],ref_array_cut[0:,15]], legend=('NN','LC','Ref'))
# rnm.plot([est_att_test_array[:,2],lc_array_cut[:,15],ref_array_cut[0:,15]], legend=('NN','LC','Ref'))
# # Plot Acc bias and Gyro Bias
# rnm.plt.figure()
# rnm.plot([est_IMU_bias_new_array[:,0],est_IMU_bias_new_array[:,1],est_IMU_bias_new_array[:,2]])

plt.show()  # Plots don't show up without this on my machine, feel free to remove it
# # Pos LSTM
# plt.figure()
# plt.grid()
# plt.title("Net position Height")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(ref_traj_llh[k, :, 2], "b", label="Reference" if k == 0 else None)
#     plt.plot(est_traj_nn_llh[k, :, 2], "r", label="Net" if k == 0 else None)
# plt.legend()
