%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%
% Copyright 2019 Mohammad Al-Sa'd
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
% Authors: Mohammad F. Al-Sa'd (mohammad.al-sad@tuni.fi)
%          Amr Mohamed         (amrm@qu.edu.qa)
%          Abdulla Al-Ali
%          Tamer Khattab
%
% The following reference should be cited whenever this script is used:
%     M. Al-Sa'd et al. "RF-based drone detection and identification using
%     deep learning approaches: an initiative towards a large open source
%     drone database", 2019.
%
% Last Modification: 12-02-2019
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

close all; clear; clc
% filepath = fileparts(pwd);
% Only when the confusion matrix gets generated
filepath = 'G:\Programing\HackathonNATO_Drones_2025\GitHubs\DroneRF-master\Python\';

%% Parameters
opt = 3;  % Change to 1, 2, or 3 to alternate between the 1st, 2nd, and 3rd DNN results respectively.
%In this MATLAB script, opt is a variable that tells the script which DNN (Deep Neural Network) results you are using. It controls two main things:

%Which columns of the CSV are used

%opt = 1 → uses columns 1–2 for true labels and 3–4 for predictions

%opt = 2 → uses columns 1–4 for true labels and 5–8 for predictions

%opt = 3 → uses columns 1–10 for true labels and 11–20 for predictions

%Which plotting function is called

%Depending on opt, the script calls plotconfusion_mod with the right slices of your CSV data.

%% Main
y = [];
for i = 1:10
    x = csvread([filepath 'Results_' num2str(opt) num2str(i) '.csv']);
    y = [y ; x];
end
%% Plotting confusion matrix
if(opt == 1)
    plotconfusion_mod(y(:,1:2)',y(:,3:4)');
elseif(opt == 2)
    plotconfusion_mod(y(:,1:4)',y(:,5:8)');
elseif(opt == 3)
    plotconfusion_mod(y(:,1:10)',y(:,11:20)');
    set(gcf,'position',[100, -100, 800, 800])
end
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));

%% Saving
Q = input('Do you want to save the results (Y/N)\n','s');
if(Q == 'y' || Q == 'Y')
    print(1,['confusion_matrix_' num2str(opt)],'-dpdf','-r512');
else
    return
end
