# =============================================================================
#  D E S C R I P T I O N
# -----------------------------------------------------------------------------
#    Filename: RNN_Kalman_Model.py
# =============================================================================
#  F U N C T I O N
# -----------------------------------------------------------------------------
#   Recurrent Neural Network Model
#
#
# -----------------------------------------------------------------------------
import torch
import torch.nn.functional as func
from torch import nn

import functions




class LC_est_KG(nn.Module):
    def __init__(
        self,
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
        cnn_hidden_channels = 32,
        cnn_out_channels = 1,
        # cnn_out_dim = 32,

    ):
        super(LC_est_KG, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.linearfc2_dim = linearfc2_dim
        self.linearfc3_dim = linearfc3_dim
        self.output_dim = output_dim
        self.Fs_imu = Fs
        self.idx_feedback_type = idx_feedback_type

        """  CNN Layer  """
        
        self.cnn = nn.Sequential(
            nn.Conv1d(12, cnn_hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_channels, cnn_out_channels, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            nn.Flatten()
        )
        # """  FC Layer1  """
        
        # self.layer1 = nn.Linear(self.input_dim, self.linearfc1_dim, bias=True)
        # self.layer1_relu = nn.ReLU()

        """  RNN Layer  """

        self.recurrent_kind = recurrent_kind
        self.rnn = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[recurrent_kind](
            self.input_dim + cnn_out_channels * self.Fs_imu - 1, self.hidden_dim, self.n_layers, batch_first=True
        )

        """  FC Layer2  """

        self.layer2 = nn.Linear(self.hidden_dim, self.linearfc2_dim, bias=True)
        # self.layer2 = nn.Linear(self.hidden_dim, self.linearfc2_dim, bias=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.Tanhsh = nn.Tanhshrink()
        self.leakyrelu = nn.LeakyReLU(0.01)
        """  FC Layer3  """
        # self.dropout = nn.Dropout(droupout_rate)
        self.layer3 = nn.Linear(self.linearfc2_dim, self.linearfc3_dim, bias=True)
        # self.layer2_relu = nn.ReLU()

        """ Linear Output Layer"""

        def print_backward(module, grad_input, grad_output):
            print("model: ", module)
            print("grad_input: ", grad_input)
            print("grad_output: ", grad_output)

        self.outputlayer = nn.Linear(self.linearfc3_dim, self.output_dim)
        # self.outputlayer.register_full_backward_hook(print_backward)
        self.biaslayer1 = nn.Linear(self.linearfc3_dim, 3, bias=False)
        self.biaslayer2 = nn.Linear(self.linearfc3_dim, 3, bias=False)

        # self.biaslayer.register_full_backward_hook(print_backward)

    def init_hidden_state(self):
        if self.recurrent_kind == "lstm":
            state1 = torch.zeros(self.n_layers, 1, self.hidden_dim)
            state2 = torch.zeros(self.n_layers, 1, self.hidden_dim)
            nn.init.xavier_normal_(state1)
            nn.init.xavier_normal_(state2)

            self.state = (state1, state2)
        else:
            self.state = torch.zeros(self.n_layers, 1, self.hidden_dim)
            nn.init.xavier_normal_(self.state)

    def init_sequence(self, Fs_meas, init_pva, init_meas, init_imu):
        self.x_old_old = torch.zeros(9) # x_t-1|t-1
        self.x_old_old_previous = torch.zeros(9) # x_t-2|t-2
        self.x_pred_old_previous = torch.zeros(9) # x_t-1|t-2
        # self.meas_old = init_meas # y_t-1
        self.meas_old = init_meas # y_t-1
        self.imu_old = init_imu
        
        self.Fs_meas = Fs_meas
        self.prev_out_old = init_pva[:9]


    def get_KG(self, features_KG, imu, dr):
        
        '''try diff'''
        dr_pos_diff = dr[1:,:3] - dr[:-1,:3]
        dr_vel_diff = dr[1:,3:6] - dr[:-1,3:6]
        '''Here can add Cbe diff'''
        # dr_att_diff = dr[1:,6:9] - dr[:-1,6:9]
        
        imu_acc_diff = imu[1:,:3] - imu[:-1,:3]
        imu_gyro_diff = imu[1:,3:] - imu[:-1,3:]

        # dr_pos_diff = functions.norm_normal_dis_2d(dr_pos_diff)
        # dr_vel_diff = functions.norm_normal_dis_2d(dr_vel_diff)
        # # dr_att_diff = functions.norm_normal_dis_2d(dr_att_diff)
        # imu_acc_diff = functions.norm_normal_dis_2d(imu_acc_diff)
        # imu_gyro_diff = functions.norm_normal_dis_2d(imu_gyro_diff)


        dr_pos_diff = func.normalize(dr_pos_diff, p=2, dim=0, eps=1e-12, out=None)
        dr_vel_diff = func.normalize(dr_vel_diff, p=2, dim=0, eps=1e-12, out=None)
        # dr_att_diff = functions.norm_normal_dis_2d(dr_att_diff)
        imu_acc_diff = func.normalize(imu_acc_diff, p=2, dim=0, eps=1e-12, out=None)
        imu_gyro_diff = func.normalize(imu_gyro_diff, p=2, dim=0, eps=1e-12, out=None)

        dr_imu = torch.cat([dr_pos_diff,dr_vel_diff,imu_acc_diff,imu_gyro_diff],dim=1)
        # dr_diff = dr
        # print(dr_diff.shape)
        cnn_out = self.cnn(dr_imu.T)

        Rnn_in = torch.cat((features_KG.reshape(-1), cnn_out.reshape(-1))).reshape(1, 1, -1)
        rnn_out, self.state = self.rnn(Rnn_in, self.state)
        rnn_out_reshape = rnn_out.reshape(1,self.hidden_dim)

        # L2_in = torch.cat((rnn_out_reshape, torch.unsqueeze(cnn_out, 0)), dim=1)
        # L2_in = torch.cat((rnn_out_reshape, cnn_out), dim=1)

        # L2_out = self.layer2(self.tanh(L2_in))
        L2_out = self.layer2(self.leakyrelu(rnn_out_reshape))

        L3_out = self.layer3(self.leakyrelu(L2_out))
        # L3_out = self.layer3(L2_out)

        # Dropout_out = self.dropout(L3_out)
        # out = self.outputlayer(self.tanh(L3_out))[0]
        out = self.outputlayer(self.leakyrelu(L3_out))

        # out[:3*6] = out[:3*6]
        # out[3*6:6*6] = out[3*6:6*6]*0.1
        # out[6*6:9*6] = out[6*6:9*6]*0.001
        
        KG = out#[0,:54]
        #accbias = out[0,54:57]
        #gyrobias = out[0,57:]
        '''to be modified, eg two layer, one for acc and one for angv'''
        # bias = self.biaslayer(self.tanh(L3_out))[0]
        accbias = self.biaslayer1(self.leakyrelu(L3_out))[0]
        gyrobias = self.biaslayer2(self.leakyrelu(L3_out))[0]
        # accbias = self.biaslayer1(L3_out)[0]
        # gyrobias = self.biaslayer2(L3_out)[0]

        if sum(abs(accbias)) > 0.5:
            accbias = accbias*0.1
        if sum(abs(gyrobias)) > 0.01:
            gyrobias = gyrobias*0.01

        # if sum(abs(accbias)) > 1:
        #     accbias = accbias*0.1
        # if sum(abs(gyrobias)) > 0.1:
        #     gyrobias = gyrobias*0.01

        # accbias = self.tanh(accbias)
        # gyrobias = self.tanh(gyrobias)

        
        
        bias = torch.cat([accbias,gyrobias])
        # print(f"Bias: {bias}")

        return KG, bias


    def forward(self, meas, dr_temp, est_C_b_e_old, imu, prev_out, dr):

        """Get Features"""

        F = functions.lc_propagate_f(dr_temp[0:3], est_C_b_e_old, imu[-1], self.Fs_meas)
        self.x_pred_old = F @ self.x_old_old
        # x_t-1|t-2                         x_t|t-1
        # self.x_error_old_previous_predict = self.x_error_old_predict
        
        f_1 = self.x_old_old - self.x_old_old_previous

        f_2 = self.x_old_old - self.x_pred_old_previous

        # f_1 = self.x_old_old

        f_3 = meas - self.meas_old
        # f_4 = torch.mean(imu[:,[0,1,5]],dim=0)
        '''could be the input to cnn, together with dr'''
        # f_4 = imu[-1] - self.imu_old
        # f_4 = self.x_old_old

        f_4 = dr_temp[0:6] - meas
        # f_5 = prev_out
        
        # f_5 = self.x_pred_old
        f_1 = func.normalize(f_1, p=2, dim=0, eps=1e-12, out=None)
        f_2 = func.normalize(f_2, p=2, dim=0, eps=1e-12, out=None)
        f_3 = func.normalize(f_3, p=2, dim=0, eps=1e-12, out=None)
        f_4 = func.normalize(f_4, p=2, dim=0, eps=1e-12, out=None)
        # f_5 = func.normalize(f_5, p=2, dim=0, eps=1e-12, out=None)

        # f_1 = functions.norm_normal_dis(f_1)
        # f_2 = functions.norm_normal_dis(f_2)
        # f_3 = functions.norm_normal_dis(f_3)
        # f_4 = functions.norm_normal_dis(f_4)



        # features_KG = torch.cat([f_1.detach(),f_2.detach(),f_3.detach(),f_5.detach()], dim=0)

        features_KG = torch.cat([f_1,f_2,f_3,f_4], dim=0)

        # KGainNet_in = KGainNet_in.detach()
        # print(KGainNet_in)
        """Kalman Gain Network Step"""
        KG, imu_error = self.get_KG(features_KG, imu, dr)

        # Reshape Kalman Gain to a Matrix
        KGain = torch.reshape(KG, (9, 6))
        """Update"""
        delta_z = dr_temp[0:6] - meas
        # if abs(sum(delta_z)) > 100:
        #     delta_z = func.normalize(delta_z, p=2, dim=0, eps=1e-12, out=None)
        Inno = KGain @ delta_z
    


        # '''another way'''
        # if self.epoch < 100:
        # self.dr_new = dr_temp - Inno
        # est_C_b_e_new = functions.euler_to_CTM(self.dr_new[6:]).T
        # else:
        #     # # '''one way'''
        dr_new_pv = dr_temp[:6] - Inno[:6]
        

        if self.idx_feedback_type == 'pvb':
            est_C_b_e_new = est_C_b_e_old
            self.dr_new = torch.cat([dr_new_pv, dr_temp[6:9], est_C_b_e_old.reshape(9)])


        if self.idx_feedback_type == 'pvab':


            if sum(abs(Inno[6:])) > 0.001:
                Inno_att_scale = Inno[6:9]*0.001
            else:
                Inno_att_scale = Inno[6:9]
            # Inno_att_scale = Inno[6:9]*1e-4
            skew_att = torch.tensor(
                [
                    [0, -Inno_att_scale[2], Inno_att_scale[1]],
                    [Inno_att_scale[2], 0, -Inno_att_scale[0]],
                    [-Inno_att_scale[1], Inno_att_scale[0], 0],
                ]
            )
            est_C_b_e_new = (torch.eye(3) - skew_att) @ est_C_b_e_old


            dr_new_att = torch.squeeze(functions.CTM_to_euler(est_C_b_e_new.T))
            self.dr_new = torch.cat([dr_new_pv, dr_new_att, est_C_b_e_new.reshape(9)])


        
        self.x_old_old_previous = self.x_old_old
        self.x_old_old = Inno
        self.x_pred_old_previous = self.x_pred_old
        self.meas_old = meas
        self.imu_old = imu[-1]
        self.prev_out_old = prev_out
        
        
        
        
        return self.dr_new, est_C_b_e_new, imu_error
    
    
    
    
    
    
    




# class LC_est_P(nn.Module):
#     def __init__(
#         self, input_dim, linearfc1_dim, hidden_dim, n_layers, linearfc2_dim, output_dim
#     ):
#         super(LC_est_P, self).__init__()
#         self.input_dim = input_dim
#         self.linearfc1_dim = linearfc1_dim
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.linearfc2_dim = linearfc2_dim
#         self.output_dim = output_dim

#         """  FC Layer1  """
#         self.layer1 = nn.Linear(self.input_dim, self.linearfc1_dim, bias=True)
#         self.layer1_relu = nn.ReLU()

#         """  RNN Layer  """

#         self.rnn_GRU = nn.GRU(
#             self.linearfc1_dim, self.hidden_dim, self.n_layers, batch_first=True
#         )
#         # self.rnn_LSTM = nn.LSTM(self.linearfc1_dim, self.hidden_dim, self.n_layers, batch_first=True)

#         """  FC Layer2  """

#         self.layer2 = nn.Linear(self.hidden_dim, self.linearfc2_dim, bias=True)
#         self.layer2_relu = nn.ReLU()

#         """  FC Layer3  """

#         self.layer3 = nn.Linear(self.linearfc2_dim, self.output_dim, bias=True)

#     def init_hidden_state(self):
#         self.state = torch.zeros(self.n_layers, 1, self.hidden_dim)
#         # self.cell = torch.randn(self.n_layers, 1, self.hidden_dim)

#     def init_sequence(self):

#         self.x_error_old = torch.zeros(9)  # (pos_3, vel_3, att_3 （9，1))
#         self.x_error_old_previous = torch.zeros(9)  # (pos_3, vel_3, att_3 （9，1))

#         """ Set measurement matrix H """
#         self.H = torch.zeros(6 * 9).reshape(6, 9)
#         self.H[0:3, 0:3] = -torch.eye(3)
#         self.H[3:6, 3:6] = -torch.eye(3)

#         """ Set measurement noise covariance matrix R """
#         gnss_pos_noise_sigma = 2.5
#         gnss_vel_noise_sigma = 0.1
#         self.R = torch.zeros(6 * 6).reshape(6, 6)
#         self.R[0, 0] = gnss_pos_noise_sigma**2
#         self.R[1, 1] = gnss_pos_noise_sigma**2
#         self.R[2, 2] = gnss_pos_noise_sigma**2
#         self.R[3, 3] = gnss_vel_noise_sigma**2
#         self.R[4, 4] = gnss_vel_noise_sigma**2
#         self.R[5, 5] = gnss_vel_noise_sigma**2

#     def get_P(self, KGainNet_in):

#         """Linear Layer 1"""
#         L1_out = self.layer1(KGainNet_in)
#         L1_relu_out = self.layer1_relu(L1_out)

#         """ GRU """
#         GRU_in = torch.zeros(1, 1, self.linearfc1_dim)
#         GRU_in[0, 0, :] = L1_relu_out
#         # self.rnn_GRU.flatten_parameters() # reduces memory usage
#         GRU_out, self.state = self.rnn_GRU(GRU_in, self.state)
#         GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

#         # ''' LSTM '''
#         # LSTM_in = torch.zeros(1, 1, self.linearfc1_dim)
#         # LSTM_in[0, 0, :] = L1_relu_out
#         # LSTM_out, (self.state, self.cell) = self.rnn_LSTM(LSTM_in, (self.state,self.cell))
#         # LSTM_out_reshape = torch.reshape(LSTM_out, (1, self.hidden_dim))

#         """ Linear Layer 2 """
#         L2_out = self.layer2(GRU_out_reshape)
#         L2_relu_out = self.layer2_relu(L2_out)

#         """ Output Layer """
#         L3_out = self.layer3(L2_relu_out)
#         return L3_out

#     def forward(self, est_pos_old, est_vel_old, est_C_b_e_old, y, imu):

#         """Get Features"""
#         y = torch.squeeze(y)
#         y_predict = torch.empty(6)
#         y_predict[0:3] = est_pos_old
#         y_predict[3:6] = est_vel_old

#         # Featture 1: x_t|t - x_t-1|t-1
#         x_f1 = self.x_error_old - self.x_error_old_previous
#         x_f1_norm = func.normalize(x_f1, p=2, dim=0, eps=1e-12, out=None)

#         # Feature 2: x_t|t-1 - x_t-1|t-1
#         # try:
#         #     x_f2 = x_predict - self.x_old
#         # except:
#         #     x_f2 = self.feature_3_init # when t=0
#         # x_f2_norm = func.normalize(x_f2)

#         # Feature 3: yt - yt-1
#         # try:
#         #     y_f3 = y - self.y_old
#         # except:
#         #     y_f3 = self.feature_3_init # when t=0
#         # y_f3_norm = func.normalize(y_f3, p=2, dim=0, eps=1e-12, out=None)

#         # # Feature 4: yt - y_t|t-1
#         # try:
#         #     y_f4 = y - y_predict
#         # except:
#         #     y_f4 = self.feature_4_init # when t=0
#         # y_f4_norm = func.normalize(y_f4, p=2, dim=0, eps=1e-12, out=None)

#         # Normalize y
#         # y_norm = func.normalize(y, p=2, dim=0, eps=1e-12, out=None);

#         # KGain Net Input
#         # KGainNet_in = torch.cat([x_f1_norm.detach(),y_f3_norm,y_f4_norm], dim=0)
#         # KGainNet_in = torch.cat([x_f1.detach(),y_f3,y_f4], dim=0)
#         # NN_in = x_f1
#         NN_in = x_f1_norm#.detach()
#         # KGainNet_in = KGainNet_in.detach()
#         # print(KGainNet_in)
#         """Kalman Gain Network Step"""
#         P_est = self.get_P(NN_in)  # (81,1)

#         # Reshape Kalman Gain to a Matrix
#         P_predict = torch.reshape(P_est, (9, 9))
#         """Update"""
#         delta_z = y - y_predict
#         if abs(sum(delta_z)) > 100:
#             delta_z = func.normalize(delta_z, p=2, dim=0, eps=1e-12, out=None)

#         KGain = (
#             P_predict
#             @ self.H.T
#             @ torch.linalg.inv(self.H @ P_predict @ self.H.T + self.R)
#         )

#         x_error = KGain @ delta_z

#         self.x_error_old_previous = self.x_error_old
#         self.x_error_old = x_error
#         self.y_old = y

#         est_pos_new = est_pos_old - x_error[0:3]
#         est_vel_new = est_vel_old - x_error[3:6]
#         skew_att = torch.tensor(
#             [
#                 [0, -x_error[8], x_error[7]],
#                 [x_error[8], 0, -x_error[6]],
#                 [-x_error[7], x_error[6], 0],
#             ]
#         )
#         est_C_b_e_new = (torch.eye(3) - skew_att) @ est_C_b_e_old

#         return est_pos_new, est_vel_new, est_C_b_e_new, [KGain, P_predict]


# class LC_est_P_Q_R(nn.Module):
#     def __init__(self, input_dim_1, linearfc1_dim_1, hidden_dim_1, n_layers_1, linearfc2_dim_1, output_dim_1,
#                        input_dim_2, linearfc1_dim_2, hidden_dim_2, n_layers_2, linearfc2_dim_2, output_dim_2,
#                        input_dim_3, linearfc1_dim_3, hidden_dim_3, n_layers_3, linearfc2_dim_3, output_dim_3):
#         super(LC_est_P_Q_R, self).__init__()
        
#         '''  GRU_1 Estimate Q  '''
#         self.input_dim_1 = input_dim_1
#         self.linearfc1_dim_1 = linearfc1_dim_1
#         self.hidden_dim_1 = hidden_dim_1
#         self.n_layers_1 = n_layers_1
#         self.linearfc2_dim_1 = linearfc2_dim_1
#         self.output_dim_1 = output_dim_1
        
#         self.layer1_1 = nn.Linear(self.input_dim_1, self.linearfc1_dim_1, bias=True)
#         self.layer1_1_relu = nn.ReLU()
#         # self.hidden_state_1 = torch.zeros(self.n_layers_1, 1, self.hidden_dim_1)
#         self.rnn_GRU_1 = nn.GRU(self.linearfc1_dim_1, self.hidden_dim_1, self.n_layers_1, batch_first=True)
#         # self.rnn_LSTM_1 = nn.LSTM(self.linearfc_dim_1, self.hidden_dim_1, self.n_layers_1, batch_first=True)
#         self.layer2_1 = nn.Linear(self.hidden_dim_1, self.linearfc2_dim_1, bias=True)
#         self.layer2_1_relu = nn.ReLU()
#         self.layer3_1 = nn.Linear(self.linearfc2_dim_1, self.output_dim_1, bias=True)

#         '''  GRU_2 Estimate P  '''
#         self.input_dim_2 = input_dim_2
#         self.linearfc1_dim_2 = linearfc1_dim_2
#         self.hidden_dim_2 = hidden_dim_2
#         self.n_layers_2 = n_layers_2
#         self.linearfc2_dim_2 = linearfc2_dim_2
#         self.output_dim_2 = output_dim_2
        
#         self.layer1_2 = nn.Linear(self.input_dim_2, self.linearfc1_dim_2, bias=True)
#         self.layer1_2_relu = nn.ReLU()
#         # self.hidden_state_2 = torch.zeros(self.n_layers_2, 1, self.hidden_dim_2)
#         self.rnn_GRU_2 = nn.GRU(self.linearfc1_dim_2, self.hidden_dim_2, self.n_layers_2, batch_first=True)
#         # self.rnn_LSTM_2 = nn.LSTM(self.linearfc_dim_2, self.hidden_dim_2, self.n_layers_2, batch_first=True)
#         self.layer2_2 = nn.Linear(self.hidden_dim_2, self.linearfc2_dim_2, bias=True)
#         self.layer2_2_relu = nn.ReLU()
#         self.layer3_2 = nn.Linear(self.linearfc2_dim_2, self.output_dim_2, bias=True)
        
#         '''  GRU_3 Estimate R  '''
#         self.input_dim_3 = input_dim_3
#         self.linearfc1_dim_3 = linearfc1_dim_3
#         self.hidden_dim_3 = hidden_dim_3
#         self.n_layers_3 = n_layers_3
#         self.linearfc2_dim_3 = linearfc2_dim_3
#         self.output_dim_3 = output_dim_3
        
#         self.layer1_3 = nn.Linear(self.input_dim_3, self.linearfc1_dim_3, bias=True)
#         self.layer1_3_relu = nn.ReLU()
#         # self.hidden_state_3 = torch.zeros(self.n_layers_3, 1, self.hidden_dim_3)
#         self.rnn_GRU_3 = nn.GRU(self.linearfc1_dim_3, self.hidden_dim_3, self.n_layers_3, batch_first=True)
#         # self.rnn_LSTM_3 = nn.LSTM(self.linearfc_dim_3, self.hidden_dim_3, self.n_layers_3, batch_first=True)
#         self.layer2_3 = nn.Linear(self.hidden_dim_3, self.linearfc2_dim_3, bias=True)
#         self.layer2_3_relu = nn.ReLU()
#         self.layer3_3 = nn.Linear(self.linearfc2_dim_3, self.output_dim_3, bias=True)

#     def init_hidden_state(self):
        
#         self.state_1 = torch.zeros(self.n_layers_1, 1, self.hidden_dim_1)
#         # self.cell_1 = torch.randn(self.n_layers, 1, self.hidden_dim_1)

#         self.state_2 = torch.zeros(self.n_layers_2, 1, self.hidden_dim_2)
#         # self.cell_2 = torch.randn(self.n_layers, 1, self.hidden_dim_2)

#         self.state_3 = torch.zeros(self.n_layers_3, 1, self.hidden_dim_3)
#         # self.cell_3 = torch.randn(self.n_layers, 1, self.hidden_dim_3)



#     def init_sequence(self):
        
#         # pos 3, vel 3, att 3 (biases are not estimated now)
#         num_state = 9
#         # GNSS measurement for LC is pos 3, vel 3
#         num_measurement = 6
        
#         # x_error_old: x_t-1|t-1
#         self.x_error_old = torch.zeros(num_state)
#         # x_error_predict: x_t|t-1
#         self.x_error_predict = torch.zeros(num_state)
#         # x_error_old_previous: x_t-2|t-2
#         self.x_error_old_previous = torch.zeros(num_state)
#         # y_old: y_t-1
#         # since it is error state kalman filter, y_t-1 is 
#         # (delta_z = gnss_pv_t-1 - dr_pv_t-1)
#         self.y_old = torch.zeros(num_measurement)
        
#         '''if we want to use QPR as features，init here'''
#         # Set initial state propagation error matrix Q

#         # Set initial Covariance matrix P
        
#         # Set initial Covariance matrix R
        
#         # Set measurement matrix H
#         self.H = torch.zeros(6*9).reshape(6,9)
#         self.H[0:3,0:3] = -torch.eye(3)
#         self.H[3:6,3:6] = -torch.eye(3)
        

#     def get_Q(self, Q_Net_in):

#         # GRU 1 get Q
#         L1_out = self.layer1_1(Q_Net_in)
#         L1_relu_out = self.layer1_1_relu(L1_out)

#         GRU_in = torch.zeros(1, 1, self.linearfc1_dim_1)
#         GRU_in[0, 0, :] = L1_relu_out
#         GRU_out, self.state_1 = self.rnn_GRU_1(GRU_in, self.state_1)
#         # GRU_out_reshape: [1,hidden_dim_1]
#         GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim_1))

#         # LSTM 1 get Q
#         # LSTM_in = torch.zeros(1, 1, self.linearfc_dim_1)
#         # LSTM_in[0, 0, :] = L1_relu_out
#         # LSTM_out, (self.state_1, self.cell_1) = self.rnn_LSTM_1(LSTM_in, (self.state_1, self.cell_1))
#         # LSTM_out_reshape = torch.reshape(LSTM_out, (1, self.hidden_dim_1))

#         L2_out = self.layer2_1(GRU_out_reshape)
#         L2_relu_out = self.layer2_1_relu(L2_out)

#         L3_out = self.layer3_1(L2_relu_out)

#         return L3_out

#     def get_P(self, P_Net_in):

#         # GRU 2 get P
#         L2_out = self.layer1_2(P_Net_in)
#         L2_relu_out = self.layer1_2_relu(L2_out)

#         GRU_in = torch.zeros(1, 1, self.linearfc1_dim_2)
#         GRU_in[0, 0, :] = L2_relu_out
#         GRU_out, self.state_2 = self.rnn_GRU_2(GRU_in, self.state_2)
#         # GRU_out_reshape: [1,hidden_dim_2]
#         GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim_2))

#         # LSTM 2 get P
#         # LSTM_in = torch.zeros(1, 1, self.linearfc_dim_2)
#         # LSTM_in[0, 0, :] = L2_relu_out
#         # LSTM_out, (self.state_2, self.cell_2) = self.rnn_LSTM_2(LSTM_in, (self.state_2, self.cell_2))
#         # LSTM_out_reshape = torch.reshape(LSTM_out, (1, self.hidden_dim_2))

#         L2_out = self.layer2_2(GRU_out_reshape)
#         L2_relu_out = self.layer2_2_relu(L2_out)

#         L3_out = self.layer3_2(L2_relu_out)
        
#         return L3_out
    
    
#     def get_R(self, R_Net_in):

#         # GRU 3 get R
#         L3_out = self.layer1_3(R_Net_in)
#         L3_relu_out = self.layer1_3_relu(L3_out)

#         GRU_in = torch.zeros(1, 1, self.linearfc1_dim_3)
#         GRU_in[0, 0, :] = L3_relu_out
#         GRU_out, self.state_3 = self.rnn_GRU_3(GRU_in, self.state_3)
#         # GRU_out_reshape: [1,hidden_dim_3]
#         GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim_3))

#         # ''' LSTM '''
#         # LSTM_in = torch.zeros(1, 1, self.linearfc_dim_3)
#         # LSTM_in[0, 0, :] = L3_relu_out
#         # LSTM_out, (self.state_3, self.cell_3) = self.rnn_LSTM_3(LSTM_in, (self.state_3, self.cell_3))
#         # LSTM_out_reshape = torch.reshape(LSTM_out, (1, self.hidden_dim_3))
        
#         L2_out = self.layer2_3(GRU_out_reshape)
#         L2_relu_out = self.layer2_3_relu(L2_out)

#         L3_out = self.layer3_3(L2_relu_out)
        
#         return L3_out
    
    
    
#     def forward(self, dr_pos, dr_vel, dr_C_b_e, gnss_pv, imu):
        
        
#         '''Get Features'''

#         F = rnm.lc_propagate_f(dr_pos, dr_C_b_e, imu)
#         # Feature 1: x_t|t-1 - x_t-1|t-1
#         self.x_error_predict = F @ self.x_error_old
#         x_f1 = self.x_error_old - self.x_error_predict
#         # x_f1 = x_f1.detach()
#         # Q_NN = self.get_Q(x_f1.detach())
#         Q_NN = self.get_Q(x_f1)
#         Q_NN = torch.reshape(Q_NN, (9, 9))
        
#         # Feature 2: x_t-1|t-1 - x_t-2|t-2
#         # since we can not get current posterior x_t|t we can only use posterior
#         # at t-1 and t-2, while it still efficient
#         x_f2 = self.x_error_old - self.x_error_old_previous
#         # x_f2norm = func.normalize(x_f2, p=2, dim=0, eps=1e-12, out=None)
#         # x_f2 = x_f2.detach()
#         # x_f2_in = torch.cat([x_f2.detach(),Q_NN.squeeze().detach()], dim=0)
#         # P_NN = self.get_P(x_f2.detach())
#         P_NN = self.get_P(x_f2)
#         P_NN = torch.reshape(P_NN, (9, 9))

#         # Feature 3: deltaz_t - deltaz_t-1
#         dr_pv = torch.zeros(6)
#         dr_pv[0:3] = dr_pos
#         dr_pv[3:6] = dr_vel

#         y_f3 = (gnss_pv - dr_pv) - self.y_old
#         # y_f3 = y_f3.detach()
#         R_NN = self.get_R(y_f3)
#         R_NN = torch.reshape(R_NN, (6, 6))
#         # Get P_predict
#         Pp = P_NN + Q_NN
#         '''Update'''
#         delta_z = gnss_pv - dr_pv
#         if abs(sum(delta_z)) > 100:
#             delta_z = func.normalize(delta_z, p=2, dim=0, eps=1e-12, out=None)
#         KGain = Pp @ self.H.T @ torch.linalg.inv(self.H @ Pp @ self.H.T + R_NN)
#         x_error = KGain @ delta_z



#         est_pos_new = dr_pos - x_error[0:3]
#         est_vel_new = dr_vel - x_error[3:6]
#         skew_att = torch.tensor([[0,-x_error[8],x_error[7]],[x_error[8],0,-x_error[6]],[-x_error[7],x_error[6],0]])
#         est_C_b_e_new = (torch.eye(3) - skew_att) @ dr_C_b_e        

#         self.x_error_old_previous = self.x_error_old
#         self.x_error_old = x_error
#         self.y_old = delta_z

#         return est_pos_new, est_vel_new, est_C_b_e_new, [KGain, P_NN, Q_NN, R_NN]
