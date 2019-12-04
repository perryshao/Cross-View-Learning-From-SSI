function read_skeletons_mat(Dataset_Folder)
% Data_Folder = '/home/data/nturgbd_skeletons/nturgb+d_skeletons'; % for ubuntu
Data_Folder = Dataset_Folder
missing_samples = 'samples_with_missing_skeletons.txt';
%  file names of NTU-RGB dataset
%  S017C003P020R002A033  S: setup No. C: camera No. P: subject No. A: action
train_n_cv = 37920;test_n_cv = 18960;
train_n_cs = 40320;test_n_cs = 16560;
train_data_cv = cell(train_n_cv,2);test_data_cv = cell(test_n_cv,2);
train_data_cs = cell(train_n_cs,2);test_data_cs = cell(test_n_cs,2);
train_camera = [2,3];test_camera = 1;
train_subjects = [1,2,4,5,8,9,13:19,25,27,28,31,34,35,38];
test_subjects =  [3,6,7,10,11,12,20:24,26,29,30,32,33,36,37,39:40];
JOINT_NUM = 25;
Data_Folder = [Data_Folder '/'];
folder_content = dir([Data_Folder,'*','.skeleton']);
sequences = size(folder_content,1);
frame_length = zeros(sequences,1);

%% read the missing skeletons
missing_samples_num = 302;
fileid = fopen(missing_samples,'r');
missing_file = cell(missing_samples_num,1);
for i = 1:missing_samples_num
	missing_file{i} = fscanf(fileid,'%s',1); % no of the recorded frames
end
missing_flag = 0;

count1 = 1; count2 = 1; count3 = 1; count4 = 1;
for n = 1:sequences
    string= [Data_Folder,folder_content(n,1).name];
    file_name = folder_content(n,1).name;
	for i = 1:missing_samples_num
		if file_name(1:end-9) == missing_file{i}
			missing_flag = 1;
			break;
		end
	end
	if missing_flag == 1;
		missing_flag = 0;
		continue;
	end
    fprintf ('Loading skeleton data...%s\n',string);
    
    bodyinfo = read_skeleton_file(string);
    framecount = size(bodyinfo,2);    
    %% First, we collect data by the cross-view protocol
    if ismember(str2num(file_name(6:8)),test_camera) % for testing
        for i = 1:framecount
            for j = 1: JOINT_NUM
                body_num = size(bodyinfo(i).bodies,2);
                for k = 1:body_num
                    test_data_cv{count1,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = bodyinfo(i).bodies(k).joints(j).x;
                    test_data_cv{count1,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = bodyinfo(i).bodies(k).joints(j).y;
                    test_data_cv{count1,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = bodyinfo(i).bodies(k).joints(j).z;
                end
				if body_num == 1 %When we observe just one skeleton, the second set is filled with zeros.
					k = 2;
					test_data_cv{count1,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = 0;
					test_data_cv{count1,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = 0;
					test_data_cv{count1,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = 0;
				end
            end
        end
        
        body_num =  size(test_data_cv{count1,1},2)/75; %redefine the body_num, because body_num is various across time steps.
		various= zeros(body_num,1);
		if body_num > 1 %When we observed two or more, we decided about which one to be the main subject and which one to be the second one, by measuring the amount of motion of their joints.
			for bi = 1:body_num
				various(bi)= sum(var(test_data_cv{count1,1}(:,(bi-1)*JOINT_NUM*3+1:bi*JOINT_NUM*3)));
			end
			[~,indx] =sort(various,'descend');
			temp = zeros(framecount,JOINT_NUM*3*2);
			temp(:,1:JOINT_NUM*3) = test_data_cv{count1,1}(:,(indx(1)-1)*JOINT_NUM*3+1:indx(1)*JOINT_NUM*3);
			temp(:,JOINT_NUM*3+1:end) = test_data_cv{count1,1}(:,(indx(2)-1)*JOINT_NUM*3+1:indx(2)*JOINT_NUM*3);
			test_data_cv{count1,1}(:,1:JOINT_NUM*3) = temp(:,1:JOINT_NUM*3);
			test_data_cv{count1,1}(:,JOINT_NUM*3+1:2*JOINT_NUM*3) = temp(:,JOINT_NUM*3+1:end);
			if body_num > 2 || size(test_data_cv{count1,1},2) > 150 
				test_data_cv{count1,1}(:,2*JOINT_NUM*3+1:end) =[];
            end	
        end
        test_data_cv{count1,2} = str2num(file_name(18:20));
        count1 = count1+1;
		frame_length(n) = framecount;
    elseif ismember(str2num(file_name(6:8)),train_camera) % for training
        for i = 1:framecount
            for j =1:JOINT_NUM
                body_num = size(bodyinfo(i).bodies,2);
                for k = 1:body_num
                    train_data_cv{count2,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = bodyinfo(i).bodies(k).joints(j).x;
                    train_data_cv{count2,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = bodyinfo(i).bodies(k).joints(j).y;
                    train_data_cv{count2,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = bodyinfo(i).bodies(k).joints(j).z;
                end
				if body_num == 1 %When we observe just one skeleton, the second set is filled with zeros.
					k = 2;
					train_data_cv{count2,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = 0;
					train_data_cv{count2,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = 0;
					train_data_cv{count2,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = 0;
				end
            end
        end
		body_num =  size(train_data_cv{count2,1},2)/75;%redefine the body_num, because body_num is various across time steps.
		various = zeros(body_num,1);
		if body_num > 1 %When we observed two or more, we decided about which one to be the main subject and which one to be the second one, by measuring the amount of motion of their joints.
			for bi = 1:body_num
				various(bi)= sum(var(train_data_cv{count2,1}(:,(bi-1)*JOINT_NUM*3+1:bi*JOINT_NUM*3)));
			end
			[~,indx] =sort(various,'descend');
			temp = zeros(framecount,JOINT_NUM*3*2);
			temp(:,1:JOINT_NUM*3) = train_data_cv{count2,1}(:,(indx(1)-1)*JOINT_NUM*3+1:indx(1)*JOINT_NUM*3);
			temp(:,JOINT_NUM*3+1:end) = train_data_cv{count2,1}(:,(indx(2)-1)*JOINT_NUM*3+1:indx(2)*JOINT_NUM*3);
			train_data_cv{count2,1}(:,1:JOINT_NUM*3) = temp(:,1:JOINT_NUM*3);
			train_data_cv{count2,1}(:,JOINT_NUM*3+1:2*JOINT_NUM*3) = temp(:,JOINT_NUM*3+1:end);
			if body_num > 2 || size(train_data_cv{count2,1},2) > 150
				train_data_cv{count2,1}(:,2*JOINT_NUM*3+1:end) =[];
			end
		end
        train_data_cv{count2,2} = str2num(file_name(18:20));
        count2 = count2+1;
		frame_length(n) = framecount;
    end
    
    %% Sencond, we collect data by the cross-subject protocol
    if ismember(str2num(file_name(10:12)), test_subjects) % for testing
        for i = 1:framecount
            for j = 1: JOINT_NUM
                body_num = size(bodyinfo(i).bodies,2);
                for k = 1:body_num
                    test_data_cs{count3,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = bodyinfo(i).bodies(k).joints(j).x;
                    test_data_cs{count3,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = bodyinfo(i).bodies(k).joints(j).y;
                    test_data_cs{count3,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = bodyinfo(i).bodies(k).joints(j).z;
                end
				if body_num == 1 %When we observe just one skeleton, the second set is filled with zeros.
					k = 2;
					test_data_cs{count3,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = 0;
					test_data_cs{count3,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = 0;
					test_data_cs{count3,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = 0;
				end
            end
        end
        
		body_num =  size(test_data_cs{count3,1},2)/75;%redefine the body_num, because body_num is various across time steps.
		various = zeros(body_num,1);
		if body_num > 1 %When we observed two or more, we decided about which one to be the main subject and which one to be the second one, by measuring the amount of motion of their joints.
			for bi = 1:body_num
				various(bi)= sum(var(test_data_cs{count3,1}(:,(bi-1)*JOINT_NUM*3+1:bi*JOINT_NUM*3)));
			end
			[~,indx] =sort(various,'descend');
			temp = zeros(framecount,JOINT_NUM*3*2);
			temp(:,1:JOINT_NUM*3) = test_data_cs{count3,1}(:,(indx(1)-1)*JOINT_NUM*3+1:indx(1)*JOINT_NUM*3);
			temp(:,JOINT_NUM*3+1:end) = test_data_cs{count3,1}(:,(indx(2)-1)*JOINT_NUM*3+1:indx(2)*JOINT_NUM*3);
			test_data_cs{count3,1}(:,1:JOINT_NUM*3) = temp(:,1:JOINT_NUM*3);
			test_data_cs{count3,1}(:,JOINT_NUM*3+1:2*JOINT_NUM*3) = temp(:,JOINT_NUM*3+1:end);
			if body_num > 2 || size(test_data_cs{count3,1},2) > 150
				test_data_cs{count3,1}(:,2*JOINT_NUM*3+1:end) =[];
			end
		end
		
        test_data_cs{count3,2} = str2num(file_name(18:20));
        count3 = count3+1;
    elseif ismember(str2num(file_name(10:12)),train_subjects) % for training
        for i = 1:framecount
            for j = 1: JOINT_NUM
                body_num = size(bodyinfo(i).bodies,2);
                for k = 1:body_num
                    train_data_cs{count4,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = bodyinfo(i).bodies(k).joints(j).x;
                    train_data_cs{count4,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = bodyinfo(i).bodies(k).joints(j).y;
                    train_data_cs{count4,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = bodyinfo(i).bodies(k).joints(j).z;
                end
				if body_num == 1 %When we observe just one skeleton, the second set is filled with zeros.
					k = 2;
					train_data_cs{count4,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+1) = 0;
					train_data_cs{count4,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+2) = 0;
					train_data_cs{count4,1}(i,(k-1)*JOINT_NUM*3+(j-1)*3+3) = 0;
                end
            end
        end
              
		body_num =  size(train_data_cs{count4,1},2)/75;%redefine the body_num, because body_num is various across time steps.
		various = zeros(body_num,1);
		if body_num > 1 %When we observed two or more, we decided about which one to be the main subject and which one to be the second one, by measuring the amount of motion of their joints.
			for bi = 1:body_num
				various(bi)= sum(var(train_data_cs{count4,1}(:,(bi-1)*JOINT_NUM*3+1:bi*JOINT_NUM*3)));
			end
			[~,indx] =sort(various,'descend');
			temp = zeros(framecount,JOINT_NUM*3*2);
			temp(:,1:JOINT_NUM*3) = train_data_cs{count4,1}(:,(indx(1)-1)*JOINT_NUM*3+1:indx(1)*JOINT_NUM*3);
			temp(:,JOINT_NUM*3+1:end) = train_data_cs{count4,1}(:,(indx(2)-1)*JOINT_NUM*3+1:indx(2)*JOINT_NUM*3);
			train_data_cs{count4,1}(:,1:JOINT_NUM*3) = temp(:,1:JOINT_NUM*3);
			train_data_cs{count4,1}(:,JOINT_NUM*3+1:2*JOINT_NUM*3) = temp(:,JOINT_NUM*3+1:end);
			if body_num > 2 || size(train_data_cs{count4,1},2) > 150
				train_data_cs{count4,1}(:,2*JOINT_NUM*3+1:end) =[];
			end
		end
        train_data_cs{count4,2} = str2num(file_name(18:20));
        count4 = count4+1;		
    end
end

% save data
test_data_cv(18933:end,:)=[];
save test_data_cv test_data_cv -v7.3;
train_data_cv(37647:end,:)=[];
save train_data_cv train_data_cv -v7.3;
test_data_cs(16488:end,:)=[];
save test_data_cs test_data_cs -v7.3;
train_data_cs(40092:end,:)=[];
save train_data_cs train_data_cs -v7.3;
save frame_length frame_length -v7.3;

