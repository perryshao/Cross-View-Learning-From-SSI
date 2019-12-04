function read_skeletons_multiscale()
%  file names of NTU-RGB dataset
%  S017C003P020R002A033  S: setup No. C: camera No. P: subject No. A: action
train_n_cv = 37646;test_n_cv = 18932;
train_n_cs = 40091;test_n_cs = 16487;
train_data_cv_scale1 = cell(train_n_cv,2);test_data_cv_scale1 = cell(test_n_cv,2);
train_data_cv_scale2 = cell(train_n_cv,2);test_data_cv_scale2 = cell(test_n_cv,2);
train_data_cv_scale3 = cell(train_n_cv,2);test_data_cv_scale3 = cell(test_n_cv,2);
train_data_cs_scale1 = cell(train_n_cs,2);test_data_cs_scale1 = cell(test_n_cs,2);
train_data_cs_scale2 = cell(train_n_cs,2);test_data_cs_scale2 = cell(test_n_cs,2);
train_data_cs_scale3 = cell(train_n_cs,2);test_data_cs_scale3 = cell(test_n_cs,2);
%% define the joint name of body 1
RANK_1 = 19 ;RKNE_1 = 18;LANK_1 = 15;LKNE_1 = 14;
LELB_1 = 6;LWRA_1 = 7;RELB_1 = 10;RWRA_1 = 11;
STRN_1 = 1;HEAD_1 = 4;RSHO_1 = 9;LSHO_1 = 5;
LFWT_1 = 13; RFWT_1 = 17;C7_1 =21;T10_1= 2;
RFIN_1 = 24;LFIN_1 = 22;RTOE_1 = 20;LTOE_1 = 16;
RFIN_A = 25;RFIN_B = 12;LFIN_A = 23;LFIN_B = 8;HEAD_A = 3;
%% define the joint name of body 2
% RANK_2 = 44 ;RKNE_2 = 43;LANK_2 = 40;LKNE_2 = 39;
% LELB_2 = 31;LWRA_2 = 32;RELB_2 = 35;RWRA_2 = 36;
% STRN_2 = 26;HEAD_2 = 29;RSHO_2 = 34;LSHO_2 = 30;
% LFWT_2 = 38; RFWT_2 = 42;C7_2 =46;T10_2= 27;
% RFIN_2 = 49;LFIN_2 = 47;RTOE_2 = 45;LTOE_2 = 41;
bodyJoints{1} = [HEAD_1 C7_1 STRN_1 RTOE_1 LTOE_1 RFIN_1 LFIN_1];  
bodyJoints{2} = [HEAD_1 C7_1 T10_1 STRN_1 RELB_1 RFIN_1 LELB_1 LFIN_1 RKNE_1 RTOE_1 LKNE_1 LTOE_1];
bodyJoints{3} = [HEAD_1 HEAD_A C7_1 T10_1 STRN_1 RSHO_1 RELB_1 RWRA_1 RFIN_A RFIN_B RFIN_1 LSHO_1 LELB_1 LWRA_1 LFIN_A LFIN_B LFIN_1 RFWT_1 RKNE_1 RANK_1  RTOE_1 LFWT_1 LKNE_1 LANK_1  LTOE_1];
body_num =2;
FullJointNum = 25;
data_folder = '/home/data/nturgbd_skeletons/ntu_data_mat/'; %for save
data_path = '/home/data/nturgbd_skeletons/ntu_data_mat/'; %for loading
% data_path = '/home/perry/ntu-works/NTURGB-D-master/Matlab/';%for loading
%% read the skeletons at 1st scale
scale = 1;
JOINT_NUM = length(bodyJoints{scale});
load ([data_path 'test_data_cv.mat'])
for n = 1:test_n_cv
    fprintf ('Loading %d-th skeleton data for test_data_cv at %d scale...\n',n,scale);
    framecount = size(test_data_cv{n,1},1);   
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                test_data_cv_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                test_data_cv_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                test_data_cv_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    test_data_cv_scale1{n,2} = test_data_cv{n,2};
end
clear test_data_cv;
test_data_cv = test_data_cv_scale1;
clear test_data_cv_scale1;
eval(['save ' data_folder 'test_data_cv_scale1' ' test_data_cv' ' -v7.3;']);
clear test_data_cv;

load ([data_path 'train_data_cv.mat'])
for n = 1:train_n_cv
    fprintf ('Loading %d-th skeleton data for train_data_cv at %d scale...\n',n,scale);
    framecount = size(train_data_cv{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                train_data_cv_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                train_data_cv_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                train_data_cv_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    train_data_cv_scale1{n,2} = train_data_cv{n,2};
end    
clear train_data_cv;
train_data_cv = train_data_cv_scale1;
clear train_data_cv_scale1;
eval(['save ' data_folder 'train_data_cv_scale1' ' train_data_cv' ' -v7.3;']);
clear train_data_cv;

load ([data_path 'test_data_cs.mat'])
for n = 1:test_n_cs
    fprintf ('Loading %d-th skeleton data for test_data_cs at %d scale...\n',n,scale);
    framecount = size(test_data_cs{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                test_data_cs_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                test_data_cs_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                test_data_cs_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    test_data_cs_scale1{n,2} = test_data_cs{n,2};
end    
clear test_data_cs;
test_data_cs = test_data_cs_scale1;
clear test_data_cs_scale1;
eval(['save ' data_folder 'test_data_cs_scale1' ' test_data_cs' ' -v7.3;']);
clear test_data_cs;

load ([data_path 'train_data_cs.mat'])
for n = 1:train_n_cs
    fprintf ('Loading %d-th skeleton data for train_n_cs at %d scale...\n',n,scale);
    framecount = size(train_data_cs{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                train_data_cs_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                train_data_cs_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                train_data_cs_scale1{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    train_data_cs_scale1{n,2} = train_data_cs{n,2};
end    
clear train_data_cs;
train_data_cs = train_data_cs_scale1;
clear train_data_cs_scale1;
eval(['save ' data_folder 'train_data_cs_scale1' ' train_data_cs' ' -v7.3;']);
clear train_data_cs;

%% read the skeletons at 2nd scale
scale = 2;
JOINT_NUM = length(bodyJoints{scale});
load ([data_path 'test_data_cv.mat'])
for n = 1:test_n_cv
    fprintf ('Loading %d-th skeleton data for test_data_cv at %d scale...\n',n,scale);
    framecount = size(test_data_cv{n,1},1);   
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                test_data_cv_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                test_data_cv_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                test_data_cv_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    test_data_cv_scale2{n,2} = test_data_cv{n,2};
end
clear test_data_cv;
test_data_cv = test_data_cv_scale2;
clear test_data_cv_scale2;
eval(['save ' data_folder 'test_data_cv_scale2' ' test_data_cv' ' -v7.3;']);
clear test_data_cv;

load ([data_path 'train_data_cv.mat'])
for n = 1:train_n_cv
    fprintf ('Loading %d-th skeleton data for train_data_cv at %d scale...\n',n,scale);
    framecount = size(train_data_cv{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                train_data_cv_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                train_data_cv_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                train_data_cv_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    train_data_cv_scale2{n,2} = train_data_cv{n,2};
end    
clear train_data_cv;
train_data_cv = train_data_cv_scale2;
clear train_data_cv_scale2;
eval(['save ' data_folder 'train_data_cv_scale2' ' train_data_cv' ' -v7.3;']);
clear train_data_cv;

load ([data_path 'test_data_cs.mat'])
for n = 1:test_n_cs
    fprintf ('Loading %d-th skeleton data for test_data_cs at %d scale...\n',n,scale);
    framecount = size(test_data_cs{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                test_data_cs_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                test_data_cs_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                test_data_cs_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    test_data_cs_scale2{n,2} = test_data_cs{n,2};
end    
clear test_data_cs;
test_data_cs = test_data_cs_scale2;
clear test_data_cs_scale2;
eval(['save ' data_folder 'test_data_cs_scale2' ' test_data_cs' ' -v7.3;']);
clear test_data_cs;

load ([data_path 'train_data_cs.mat'])
for n = 1:train_n_cs
    fprintf ('Loading %d-th skeleton data for train_n_cs at %d scale...\n',n,scale);
    framecount = size(train_data_cs{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                train_data_cs_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                train_data_cs_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                train_data_cs_scale2{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    train_data_cs_scale2{n,2} = train_data_cs{n,2};
end    
clear train_data_cs;
train_data_cs = train_data_cs_scale2;
clear train_data_cs_scale2;
eval(['save ' data_folder 'train_data_cs_scale2' ' train_data_cs' ' -v7.3;']);
clear train_data_cs;
%% read the skeletons at 3rd scale
scale = 3;
JOINT_NUM = length(bodyJoints{scale});
load ([data_path 'test_data_cv.mat'])
for n = 1:test_n_cv
    fprintf ('Loading %d-th skeleton data for test_data_cv at %d scale...\n',n,scale);
    framecount = size(test_data_cv{n,1},1);   
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                test_data_cv_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                test_data_cv_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                test_data_cv_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = test_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    test_data_cv_scale3{n,2} = test_data_cv{n,2};
end
clear test_data_cv;
test_data_cv = test_data_cv_scale3;
clear test_data_cv_scale3;
eval(['save ' data_folder 'test_data_cv_scale3' ' test_data_cv' ' -v7.3;']);
clear test_data_cv;

load ([data_path 'train_data_cv.mat'])
for n = 1:train_n_cv
    fprintf ('Loading %d-th skeleton data for train_data_cv at %d scale...\n',n,scale);
    framecount = size(train_data_cv{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                train_data_cv_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                train_data_cv_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                train_data_cv_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = train_data_cv{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    train_data_cv_scale3{n,2} = train_data_cv{n,2};
end    
clear train_data_cv;
train_data_cv = train_data_cv_scale3;
clear train_data_cv_scale3;
eval(['save ' data_folder 'train_data_cv_scale3' ' train_data_cv' ' -v7.3;']);
clear train_data_cv;

load ([data_path 'test_data_cs.mat'])
for n = 1:test_n_cs
    fprintf ('Loading %d-th skeleton data for test_data_cs at %d scale...\n',n,scale);
    framecount = size(test_data_cs{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                test_data_cs_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                test_data_cs_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                test_data_cs_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = test_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    test_data_cs_scale3{n,2} = test_data_cs{n,2};
end    
clear test_data_cs;
test_data_cs = test_data_cs_scale3;
clear test_data_cs_scale3;
eval(['save ' data_folder 'test_data_cs_scale3' ' test_data_cs' ' -v7.3;']);
clear test_data_cs;

load ([data_path 'train_data_cs.mat'])
for n = 1:train_n_cs
    fprintf ('Loading %d-th skeleton data for train_n_cs at %d scale...\n',n,scale);
    framecount = size(train_data_cs{n,1},1);    
    for i = 1:framecount
        for j = 1:JOINT_NUM
            for k = 1:body_num
                train_data_cs_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+1);
                train_data_cs_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+2);
                train_data_cs_scale3{n,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = train_data_cs{n,1}(i,(k-1)*FullJointNum*3+(bodyJoints{scale}(j)-1)*3+3);
            end
        end
    end
    train_data_cs_scale3{n,2} = train_data_cs{n,2};
end    
clear train_data_cs;
train_data_cs = train_data_cs_scale3;
clear train_data_cs_scale3;
eval(['save ' data_folder 'train_data_cs_scale3' ' train_data_cs' ' -v7.3;']);
clear train_data_cs;

