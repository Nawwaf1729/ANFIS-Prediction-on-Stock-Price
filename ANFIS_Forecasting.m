clc;clear;close all;tic
% Program for Time Series Prediction

%% Load & Plot Dataset

% Load data
kel7_data = readtable('C:\Users\User\Documents\MATLAB\Data Harga Saham ASII.xlsx');

time = kel7_data.Date;
x = kel7_data.Close_Price;

% Plot data ASII
figure
plot(time, x)
title('Closing Price ASII')
xlabel('Time')
ylabel('Price')

%% Pre-Processing Data

% Preparing input-output data matrix
for t = 100:931
    data(t-99,:) = [x(t-15) x(t-10) x(t-5) x(t) x(t+5)];
end

trndata = data(1:600,:);
valdata = data(601:end,:);

%% ANFIS Training and Testing

% Generating FIS
genfis_opt = genfisOptions('GridPartition');
genfis_opt.InputMembershipFunctionType = 'gauss2mf';

fis = genfis(trndata(:,1:end-1), trndata(:,end), genfis_opt);

% Defining training options for ANFIS
options = anfisOptions('InitialFIS',fis,'ValidationData',valdata,'EpochNumber',50);

% Training ANFIS
[fis1,error1,ss,fis2,error2] = anfis(trndata,options);

% Evaluating ANFIS for 5 samples prediction
anfis_output = evalfis(fis2,[trndata(:,1:4);valdata(:,1:4)]);

%% Plotting

% Plotting input and output predicted series
index = 105:936;
index1 = 932:936;

%Plot Actual vs ANFIS Prediction
figure
plot(time(index), [x(index) anfis_output])
xlabel('Time')
legend('Actual ASII','ANFIS Prediction')
title('ASII and ANFIS Prediction')

%Plot Next 5 Samples Prediction
figure
plot(time(index1+1), [x(index1) anfis_output(end-4:end)])
hold on;
plot(time(index1+1), [x(index1) anfis_output(end-4:end)], 'o','MarkerFaceColor','k')
xticks(time(index1+1))
xlabel('Time')
legend('Actual ASII','ANFIS Prediction')
title('Next 5 Samples Prediction and Actual ASII')

% Plotting Training and Validation Error
figure
plot([error1 error2]); hold on; plot([error1 error2], 'o')
legend('Training error','Validation error')
xlabel('Epoch'); ylabel('RMS Error')
title('Error Plots')

% Finding Errors (Difference in prediction)
diff = x(index) - anfis_output;
figure
plot(time(index),diff)
xlabel('Time'); title('Prediction Errors')
title('Difference in Prediction')

% Compute RMSE for training and validation data
rmse_training = sqrt(mean((trndata(:,end) - anfis_output(1:600)).^2));
rmse_validation = sqrt(mean((valdata(:,end) - anfis_output(601:end)).^2));

disp(['Training RMSE: ', num2str(rmse_training)]);
disp(['Validation RMSE: ', num2str(rmse_validation)]);

% Plot RMSE bar
figure
bar([rmse_training rmse_validation])
set(gca, 'xticklabel',{'Training RMSE', 'Validation RMSE'})
ylabel('RMSE')
title('RMSE for Training and Validation Data')
