a_num = 1;c_num = 1;d_num = 1;p_num = 1;
ACT_NUM=0:9;CAMERA_NUM=1:2;DIRECTION_NUM=1:8;PEOPLE_NUM=1:40;
%action1-6 camera1-2 direction1-8 people1-40
JOINT_NUM = 25;
SKL_TMP=zeros(JOINT_NUM,7,1);
SAVEDIR='SKL_DATA'
if ~exist('SKL_DATA','dir')
    mkdir(SAVEDIR);
end
for p_num = PEOPLE_NUM
    for a_num = ACT_NUM
        for c_num = CAMERA_NUM
            for d_num = DIRECTION_NUM
                path_readtxt = strcat('a',num2str(a_num),'_d',num2str(d_num),'_p',num2str(p_num,'%02d'),'_c',num2str(c_num));
                path_savemat = strcat('a',num2str(a_num),'_d',num2str(d_num),'_p',num2str(p_num,'%02d'),'_c',num2str(c_num));
                %% skeleton extract
                [f,message] = fopen(strcat(path_readtxt,'_skeleton.txt'),'r');
                SKL = [];
                i=1;j=1;k=1;
                while ~feof(f)
                    read_info = fgetl(f);
                    if size(strfind(read_info,'#'),2)~=0;
                    else
                        data = str2num(read_info);
                        SKL_TMP(i,1:3,1)=data(1,1:3);
                        data = str2num(fgetl(f));
                        SKL_TMP(i,4:5,1)=data(1,1:2);
                        data = str2num(fgetl(f));
                        SKL_TMP(i,6:7,1)=data(1,1:2);
                        i=i+1;
                        if i >= 26
                            i = 1;k=k+1;
                            SKL=cat(3,SKL,SKL_TMP);
                        end
                    end
                end
                fclose(f);               
                save(strcat(SAVEDIR,'\',path_readtxt,'_skl.mat'),'SKL');             
            end
        end
    end
end