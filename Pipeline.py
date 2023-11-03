# =============================================================================
#  D E S C R I P T I O N
# -----------------------------------------------------------------------------
#    Filename: Pipeline.py
# =============================================================================
#  F U N C T I O N
# -----------------------------------------------------------------------------
#   Training Pipeline
#
#
# -----------------------------------------------------------------------------
import torch

import functions


class Pipeline_LC:
    def __init__(self, net, num_epochs, learning_rate, weight_decay, loss_fn, scheduler_generator,nograd, lossweight_coeff, train_gnssgap):

        self.net = net
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        # Adam
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.nograd = nograd
        self.lcoe_p = float(lossweight_coeff[0])
        self.lcoe_v = float(lossweight_coeff[1])
        self.lcoe_a = float(lossweight_coeff[2])

        self.get_x_init_error_free_batched = torch.vmap(self.get_x_init_error_free)

        # self.optimizer = torch.optim.Adadelta(
        #     self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
        # )
        # SGD
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        if scheduler_generator is not None:
            self.scheduler = scheduler_generator(self.optimizer)
        else:
            self.scheduler = None
        
        self.train_gnssgap = train_gnssgap


    def get_x_init_error_free(self,ref):
        
        
        C_b_n = functions.euler_to_CTM(ref[18:21]).T
        est_pos_old, est_vel_old, est_C_b_e_old = functions.geo_ned2ecef(ref[12:15],ref[9:12],C_b_n)
        est_att_old = torch.squeeze(functions.CTM_to_euler(est_C_b_e_old.T))

        # est_IMU_bias_old = torch.zeros(6)
        
        return est_pos_old, est_vel_old, est_C_b_e_old, est_att_old#, est_IMU_bias_old

    def train_lc(
        self, train_loader, train_dataset, val_loader, val_dataset, Fs, Fs_meas
    ):

        for epoch in range(self.num_epochs):

            self.net.epoch = epoch
            self.net.train()

            X, y, imu_meas = next(iter(train_loader))

            batch_size = X.shape[0]

            '''Here, we could use initialization stategy to init hidden state'''
            self.net.init_hidden_state(batch_size)

            # pos3 vel3 att3 Cbe9
            output_all = torch.zeros(batch_size, train_dataset.time_step * Fs, 18)
            imu_error = torch.zeros(batch_size, 6)

            # # apply a random Gaussian error to the ref as init pva
            # est_pos_old, est_vel_old, est_C_b_e_old = self.get_x_init(
            #     train_dataset.targets[i, 0, :]
            # )
            # Use Error free init pva
            est_pos_init, est_vel_init, est_C_b_e_iter, est_att_init = self.get_x_init_error_free_batched(y[:, 0, :])

            '''[pos, vel ,att[b_e]]'''
            last_output_train = torch.cat([est_pos_init, est_vel_init, est_att_init, est_C_b_e_iter.flatten(1)], dim=1)

            self.net.init_sequence(Fs_meas, last_output_train, X[:, 0, :], imu_meas[:, 0, -1, :])

            timesteps = X.shape[1] - 1 # skip first one
            for t in range(timesteps):
                if self.nograd:
                    with torch.no_grad():
                        dr_temp, est_C_b_e_iter = functions.dead_reckoning_ecef_batched(
                            Fs,
                            last_output_train,
                            est_C_b_e_iter,
                            imu_meas[:, t],
                            imu_error,
                        )
                else:
                    dr_temp, est_C_b_e_iter = functions.dead_reckoning_ecef_batched(
                        Fs,
                        last_output_train,
                        est_C_b_e_iter,
                        imu_meas[:, t],
                        imu_error,
                    )

                if self.train_gnssgap:

                    if t < 10 or t > 15:
                        # if t < -1:
                        output_train, est_C_b_e_iter, imu_error = self.net(
                            X[:, t + 1, :],
                            dr_temp[:, -1],
                            est_C_b_e_iter,
                            imu_meas[:, t],
                            last_output_train,
                            dr_temp[:, -Fs:],
                        )

                        output_all[:, Fs * t + 1: Fs * t + Fs] = dr_temp[:, 1:-1]
                        output_all[:, Fs * t + Fs] = output_train
                        last_output_train = output_train
                    else:
                        output_all[Fs * t + 1: Fs * t + Fs] = dr_temp[:, 1:-1]
                        output_all[Fs * t + Fs] = dr_temp[:, -1]
                        last_output_train = dr_temp[:, -1]
                else:
                    output_train, est_C_b_e_iter, imu_error = self.net(
                        X[:, t + 1, :],
                        dr_temp[:, -1],
                        est_C_b_e_iter,
                        imu_meas[:, t],
                        last_output_train,
                        dr_temp[:, -Fs:],
                    )

                    output_all[:, Fs * t + 1: Fs * t + Fs] = dr_temp[:, 1:-1]
                    output_all[:, Fs * t + Fs] = output_train
                    last_output_train = output_train

            loss1 = self.loss_fn(output_all[:, Fs:t * Fs + 1, 0:3], y[:, Fs:t * Fs + 1, 0:3])
            loss2 = self.loss_fn(output_all[:, Fs:t * Fs + 1, 3:6], y[:, Fs:t * Fs + 1, 3:6])
            loss3 = self.loss_fn(1e2 * output_all[:, Fs:t * Fs + 1, 9:18], 1e2 * y[:, Fs:t * Fs + 1, 24:33])

            self.optimizer.zero_grad()
            loss_mean_train = loss1 + loss2 + loss3
            loss_mean_train.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(),1)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()


            # Epoch_Loss_mean = loss_pos_mean + loss_vel_mean + loss_att_mean
            # Epoch_Loss_mean.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            # self.optimizer.step()
            # self.scheduler.step()
            # MSE_train_epoch[epoch] = Epoch_Loss_mean

            print(f"Epoch {epoch}")
            print(f"MSE Loss in train data set is {float(loss_mean_train):f}, IMU error is {imu_error}")
            # print(f'MSE Loss in val data set is {float(Loss_val):f}')
            # save NN when reach minimum validation loss
            # if Loss_val < min_Loss_val:
            #     min_Loss_val = Loss_val
            #     optimal_epoch = epoch
            #     torch.save(self.net,self.out_nn_name)

            # print(f'Optimal NN is at Epoch {optimal_epoch}, loss is {float(min_Loss_val):f}')

        # writer.add_hparams(
        #     {
        #         "vel_coef": vel_coef,
        #         "att_coef": att_coef,
        #         "lr": self.optimizer.param_groups[0]["lr"],
        #         "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
        #         "time_step": train_timestep,
        #         "batch_size": self.batch_size,
        #     },
        #     {
        #         name: np.array([metric.detach().cpu()])  # ignore the typing error
        #         for name, metric in writer_metrics.items()
        #     },
        # )
        # return (
        #     est_pos_new,
        #     est_vel_new,
        #     est_att_new,
        #     MSE_train_epoch,
        #     MSE_val_epoch,
        #     check_list
        # )  

    def test_lc(self, test_loader, test_dataset, Fs, Fs_meas):

        # time_step_test = dataset.features.shape[0]
        # test_features = torch.squeeze(dataset.features)
        # test_targets = torch.squeeze(dataset.targets)
        # imu_meas_test = torch.squeeze(dataset.imu_meas)

        # est_pos_new = torch.zeros(time_step_test, 3)
        # est_vel_new = torch.zeros(time_step_test, 3)
        # est_C_b_e_new = torch.zeros((time_step_test, 3, 3))
        # est_att_new = torch.zeros(time_step_test, 3)

        # est_IMU_bias_new = torch.zeros(time_step_test, 6)

        predict_traj = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 18))
        ref_traj = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 33))
        
        predict_traj_llh = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 6))
        ref_traj_llh = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 6))        
        bias_history = torch.zeros(test_dataset.num_traj, test_dataset.time_step, 6)
        

        dt = 1 / Fs

        self.net.init_hidden_state(1)

        # init features
        # self.net.init_sequence()

        # # apply a random Gaussian error to the ref as init pva
        # est_pos_old, est_vel_old, est_C_b_e_old = self.get_x_init(
        #     test_targets[0, :]
        # )
        # Use Error free init pva
        # est_pos_old, est_vel_old, est_C_b_e_old = self.get_x_init_error_free(
        #     test_targets[0, :]
        # )

        for i, (X, y, imu_meas) in enumerate(test_loader):
            
            imu_error = torch.zeros(6)
            
            est_pos_init, est_vel_init, est_C_b_e_iter, est_att_init = self.get_x_init_error_free(y[0, 0])
            # _,_,est_att_init= functions.ecef2geo_ned(est_pos_init, est_vel_init, est_C_b_e_iter)
            
            predict_traj[i, 0] = torch.cat([est_pos_init, est_vel_init, est_att_init, est_C_b_e_iter.reshape(9)])
            imu_meas = imu_meas.reshape(1, -1, 6)

            # init features
            self.net.init_sequence(Fs_meas, predict_traj[i, 0].unsqueeze(0), X[0, 0].unsqueeze(0), imu_meas[0, 0, -1].unsqueeze(0))
            
            for t in range(predict_traj.shape[1] - 1):
                
                # if t % Fs == 0 and t != 0:
                # if t % Fs == 0 and t != 0 and t!= 100 and t!= 110 and t!= 120 and t!= 130 and t!= 140 and t!= 150 :
                if t % Fs == 0 and t != 0 and t!= 1000 and t!= 1010 and t!= 1020 and t!= 1030 and t!= 1040 and t!= 1050 and t!= 1060 and t!= 1070 and t!= 1080 and t!= 1090 and t!= 1100 and t!= 1110:

                # if t % Fs == 0 and t != 0 and t!= 100 and t!= 110 and t!= 120 and t!= 130 and t!= 140 and t!= 150 and t!= 160 and t!= 170 and t!= 180 and t!= 190 and t!= 200 and t!= 210 :

                # if t <-1:
                    predict_traj[i, t, :], est_C_b_e_iter, imu_error = self.net.forward_unbatched(
                        X[0, int(t / Fs), :],
                        predict_traj[i, t],
                        est_C_b_e_iter,
                        imu_meas[0,t-Fs:t],
                        predict_traj[i, t - 1],
                        predict_traj[i, t - Fs : t],
                    )
                    
                    bias_history[i, int(t / Fs), :] = imu_error
                
                predict_traj[i, t + 1], est_C_b_e_iter = functions.dead_reckoning_ecef_test(
                    Fs,
                    predict_traj[i, t],
                    est_C_b_e_iter,
                    imu_meas[0, t],
                    imu_error,
                )
            
            ref_traj[i, :, :] = y
            
        with torch.no_grad():
            for i in range(ref_traj.shape[0]):
                for j in range(ref_traj.shape[1]):
                    predict_traj_llh[i,j,:3]  = torch.squeeze(functions.ecef2geo_ned(predict_traj[i,j,:3],predict_traj[i,j,3:6])[0])
                    predict_traj_llh[i,j,3:6] = torch.squeeze(functions.ecef2geo_ned(predict_traj[i,j,:3],predict_traj[i,j,3:6])[1])
                    ref_traj_llh[i,j,:3]  = torch.squeeze(functions.ecef2geo_ned(ref_traj[i,j,:3],ref_traj[i,j,3:6])[0])
                    ref_traj_llh[i,j,3:6] = torch.squeeze(functions.ecef2geo_ned(ref_traj[i,j,:3],ref_traj[i,j,3:6])[1])
                
        return predict_traj, ref_traj, bias_history, predict_traj_llh, ref_traj_llh

                
    def GnssInsLooseCoupling(self, datatype, test_loader, test_dataset, Fs, Fs_meas):

        dt_imu = 1/Fs
        print('Datatype is:' + datatype)
        if  datatype == 'bias' :
            accel_noise_PSD_x = 0#100 * functions.micro_g_to_meters_per_second_squared ** 2
            accel_noise_PSD_y = 0#100 * functions.micro_g_to_meters_per_second_squared ** 2
            accel_noise_PSD_z = 0#100 * functions.micro_g_to_meters_per_second_squared ** 2
            # Acc bias variation
            # (m^2 s^-5)
            accel_bias_PSD_x = 3e-16
            accel_bias_PSD_y = 3e-16
            accel_bias_PSD_z = 3e-16
            # Angular random walk
            # (deg^2 per hour, converted to rad^2/s)
            gyro_noise_PSD_x = 0#(0.01 * functions.deg_to_rad / 60) ** 2#3e-15**2  # (0.01 * gim.deg_to_rad / 60) ** 2
            gyro_noise_PSD_y = 0#(0.01 * functions.deg_to_rad / 60) ** 2#3e-15**2  # 
            gyro_noise_PSD_z = 0#(0.01 * functions.deg_to_rad / 60) ** 2#3e-15**2  # 
            # Gyro bias variation
            # (rad^2 s^-3)
            gyro_bias_PSD_x = 2e-16
            gyro_bias_PSD_y = 2e-16
            gyro_bias_PSD_z = 2e-16
            
        if  datatype == 'MEMS' :
            accel_noise_PSD_x = 300 * functions.micro_g_to_meters_per_second_squared ** 2
            accel_noise_PSD_y = 300 * functions.micro_g_to_meters_per_second_squared ** 2
            accel_noise_PSD_z = 300 * functions.micro_g_to_meters_per_second_squared ** 2
            # Acc bias variation
            # (m^2 s^-5)
            accel_bias_PSD_x = 3e-6
            accel_bias_PSD_y = 3e-6
            accel_bias_PSD_z = 3e-6
            # Angular random walk
            # (deg^2 per hour, converted to rad^2/s)
            gyro_noise_PSD_x = (0.05 * functions.deg_to_rad / 60) ** 2#3e-15**2  # (0.01 * gim.deg_to_rad / 60) ** 2
            gyro_noise_PSD_y = (0.05 * functions.deg_to_rad / 60) ** 2#3e-15**2  # 
            gyro_noise_PSD_z = (0.05 * functions.deg_to_rad / 60) ** 2#3e-15**2  # 
            # Gyro bias variation
            # (rad^2 s^-3)
            gyro_bias_PSD_x = 2e-8
            gyro_bias_PSD_y = 2e-8
            gyro_bias_PSD_z = 2e-8
            
        if  datatype == 'real' :
            accel_noise_PSD_x = 1000 * functions.micro_g_to_meters_per_second_squared ** 2
            accel_noise_PSD_y = 1000 * functions.micro_g_to_meters_per_second_squared ** 2
            accel_noise_PSD_z = 1000 * functions.micro_g_to_meters_per_second_squared ** 2
            # Acc bias variation
            # (m^2 s^-5)
            accel_bias_PSD_x = 3e-4
            accel_bias_PSD_y = 3e-4
            accel_bias_PSD_z = 3e-4
            # Angular random walk
            # (deg^2 per hour, converted to rad^2/s)
            gyro_noise_PSD_x = (0.1 * functions.deg_to_rad / 60) ** 2#3e-15**2  # (0.01 * gim.deg_to_rad / 60) ** 2
            gyro_noise_PSD_y = (0.1 * functions.deg_to_rad / 60) ** 2#3e-15**2  # 
            gyro_noise_PSD_z = (0.1 * functions.deg_to_rad / 60) ** 2#3e-15**2  # 
            # Gyro bias variation
            # (rad^2 s^-3)
            gyro_bias_PSD_x = 2e-6
            gyro_bias_PSD_y = 2e-6
            gyro_bias_PSD_z = 2e-6

        if  datatype == 'GIVE' :
            accel_noise_PSD_x = 5.835394746507365e-04 * 1e-2
            accel_noise_PSD_y = 5.932170844681660e-04 * 1e-2
            accel_noise_PSD_z = 0.001470026786190 * 1e-2
            # Acc bias variation
            # (m^2 s^-5)
            accel_bias_PSD_x = 9.051543482378678e-05 * 1e-2
            accel_bias_PSD_y = 9.391773577249606e-05 * 1e-2
            accel_bias_PSD_z = 6.115995864632839e-05 * 1e-2
            # Angular random walk
            # (deg^2 per hour, converted to rad^2/s)
            gyro_noise_PSD_x = 6.43396e-05 * 1e-2
            gyro_noise_PSD_y = 7.35175e-05 * 1e-2# rad/s^1.5
            gyro_noise_PSD_z = 5.93654e-05 * 1e-2# rad/s^1.5
            # Gyro bias variation
            # (rad^2 s^-3)
            gyro_bias_PSD_x = 3.01668e-06 * 1e-2
            gyro_bias_PSD_y = 4.15136e-06 * 1e-2
            gyro_bias_PSD_z = 1.00871e-05 * 1e-2


        q_psd_tensor = torch.tensor([
            accel_noise_PSD_x,
            accel_noise_PSD_y,
            accel_noise_PSD_z,
            accel_bias_PSD_x,
            accel_bias_PSD_y,
            accel_bias_PSD_z,
            gyro_noise_PSD_x,
            gyro_noise_PSD_y,
            gyro_noise_PSD_z,
            gyro_bias_PSD_x,
            gyro_bias_PSD_y,
            gyro_bias_PSD_z,
        ])
        
        accel_noise_PSD = torch.diag(q_psd_tensor[:3])
        accel_bias_PSD = torch.diag(q_psd_tensor[3:6])
        gyro_noise_PSD = torch.diag(q_psd_tensor[6:9])
        gyro_bias_PSD = torch.diag(q_psd_tensor[9:])
    
        Q = torch.zeros((15,15))
    
        # Q(1:3,1:3)     = 0
        Q[3:6, 3:6] = accel_noise_PSD * dt_imu
        Q[6:9, 6:9] = gyro_noise_PSD * dt_imu
        Q[9:12, 9:12] = accel_bias_PSD * dt_imu
        Q[12:15, 12:15] = gyro_bias_PSD * dt_imu
    
        
        
        """Set Pr and Pr rate measurement noise SD"""
        gnss_pos_noise_sigma = 1
        gnss_vel_noise_sigma = 0.1
    
    
        R = torch.zeros((6,6))
        R[0, 0] = gnss_pos_noise_sigma**2
        R[1, 1] = gnss_pos_noise_sigma**2
        R[2, 2] = gnss_pos_noise_sigma**2
        R[3, 3] = gnss_vel_noise_sigma**2
        R[4, 4] = gnss_vel_noise_sigma**2
        R[5, 5] = gnss_vel_noise_sigma**2
    
    
        """Test"""
        # Initial position uncertainty per axis (m)
        init_pos_unc_x = 3
        init_pos_unc_y = 3
        init_pos_unc_z = 3
        # Initial velocity uncertainty per axis (m/s)
        init_vel_unc_x = 1
        init_vel_unc_y = 1
        init_vel_unc_z = 1
        # Initial attitude uncertainty per axis (deg, converted to rad)
        init_att_unc_x = (5 * 3.14/180) ** 2
        init_att_unc_y = (5 * 3.14/180) ** 2
        init_att_unc_z = (5 * 3.14/180) ** 2
        # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
        init_b_a_unc_x = 0.2 ** 2
        init_b_a_unc_y = 0.2 ** 2
        init_b_a_unc_z = 0.2 ** 2
        # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
        init_b_g_unc_x = 0.02 ** 2
        init_b_g_unc_y = 0.02 ** 2
        init_b_g_unc_z = 0.02 ** 2
        
        p_unc_tensor = torch.tensor([
            init_pos_unc_x,
            init_pos_unc_y,
            init_pos_unc_z,
            init_vel_unc_x,
            init_vel_unc_y,
            init_vel_unc_z,
            init_att_unc_x,
            init_att_unc_y,
            init_att_unc_z,
            init_b_a_unc_x,
            init_b_a_unc_y,
            init_b_a_unc_z,
            init_b_g_unc_x,
            init_b_g_unc_y,
            init_b_g_unc_z,
        ])
    
        P = torch.zeros((15,15))
        P[0:3, 0:3] = torch.diag(p_unc_tensor[:3]) ** 2
        P[3:6, 3:6] = torch.diag(p_unc_tensor[3:6]) ** 2
        P[6:9, 6:9] = torch.diag(p_unc_tensor[6:9]) ** 2
        P[9:12, 9:12] = torch.diag(p_unc_tensor[9:12]) ** 2
        P[12:15, 12:15] = torch.diag(p_unc_tensor[12:]) ** 2



        predict_traj = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 18))
        P_est = torch.zeros((test_dataset.num_traj, test_dataset.time_step, 15, 15))
        ref_traj = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 33))
        
        predict_traj_llh = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 6))
        ref_traj_llh = torch.zeros((test_dataset.num_traj, test_dataset.time_step * Fs, 6))        
        bias_history = torch.zeros(test_dataset.num_traj, test_dataset.time_step, 6)
        

        dt = 1 / Fs


        for i, (X, y, imu_meas) in enumerate(test_loader):
            
            imu_error = torch.zeros(6)
            
            est_pos_init, est_vel_init, est_C_b_e_iter, est_att_init = self.get_x_init_error_free(y[0, 0])
            # _,_,est_att_init= functions.ecef2geo_ned(est_pos_init, est_vel_init, est_C_b_e_iter)
            
            predict_traj[i, 0] = torch.cat([est_pos_init, est_vel_init, est_att_init, est_C_b_e_iter.reshape(9)])
            imu_meas = imu_meas.reshape(1, -1, 6)

            # init features

            P_est[i, 0] = P

            
            for t in range(predict_traj.shape[1] - 1):
                
                # if t % Fs == 0 and t != 0 and t!= 100 and t!= 110 and t!= 120 and t!= 130 and t!= 140 and t!= 150:
                if t % Fs == 0 and t != 0 and t!= 1000 and t!= 1010 and t!= 1020 and t!= 1030 and t!= 1040 and t!= 1050:
                # if t <-1:
                # if t % Fs == 0 and t != 0:
                    predict_traj[i, t, :], est_C_b_e_iter, imu_error, P_est[i, int(t / Fs)] = functions.Model_LC(
                        X[0, int(t / Fs), :],
                        predict_traj[i, t],
                        est_C_b_e_iter,
                        imu_meas[0,t],
                        imu_error,
                        P_est[i, int(t / Fs) - 1],
                        Q,
                        R,
                        Fs_meas,
                    )
                    
                    bias_history[i, int(t / Fs), :] = imu_error
                
                predict_traj[i, t + 1], est_C_b_e_iter = functions.dead_reckoning_ecef_test(
                    Fs,
                    predict_traj[i, t],
                    est_C_b_e_iter,
                    imu_meas[0, t],
                    imu_error,
                )
            
            ref_traj[i, :, :] = y
            
        with torch.no_grad():
            for i in range(ref_traj.shape[0]):
                for j in range(ref_traj.shape[1]):
                    predict_traj_llh[i,j,:3]  = torch.squeeze(functions.ecef2geo_ned(predict_traj[i,j,:3],predict_traj[i,j,3:6])[0])
                    predict_traj_llh[i,j,3:6] = torch.squeeze(functions.ecef2geo_ned(predict_traj[i,j,:3],predict_traj[i,j,3:6])[1])
                    ref_traj_llh[i,j,:3]  = torch.squeeze(functions.ecef2geo_ned(ref_traj[i,j,:3],ref_traj[i,j,3:6])[0])
                    ref_traj_llh[i,j,3:6] = torch.squeeze(functions.ecef2geo_ned(ref_traj[i,j,:3],ref_traj[i,j,3:6])[1])
                
        return predict_traj, ref_traj, bias_history, predict_traj_llh, ref_traj_llh

class Pipeline_TC:
    def __init__(
        self,
        net,
        num_epochs,
        learning_rate,
        weight_decay,
        batch_size,
        time_step,
        loss_fn,
    ):
        self.batch_size = batch_size
        self.time_step = time_step
        self.net = net
        self.num_epochs = num_epochs
        # MSE LOSS Function
        self.loss_fn = loss_fn
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
