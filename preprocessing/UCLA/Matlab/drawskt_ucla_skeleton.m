function drawskt_ucla_skeleton()
%% global parameters
% sf: start frame
% ef: ending frame
J=[20     19     18     17    1    13     14   15   1    2   3    3     5    6   7   3    9    10   11 ;
   19     18     17     1    13    14     15   16   2    3   4    5     6    7   8   9    10   11   12 ];
train_samples = 937;
test_samples = 463;
joint_number = 20;
actions = 10;
filepath = 'C:\Users\perry\Desktop\ntu3daction\';
%%%%%%%%%%%%%%%%%%%%%--view_1--%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% draw the view_1
frames = 16; % full length = 32
samples = 138;%238
filename = 'Train_Raw_cv3.h5';
labels_dataset = 'y_train';
dataset = 'x_train';
%% to save the labels
% labels1 = h5read([filepath filename],['/' labels_dataset],[1,1],[actions,train_samples]);
% save labels1 labels1;
% frame_length =  h5read([filepath 'frame_length.h5'],['/' 'frame_length'],[1],[1400]);
% save frame_length frame_length
figure(1)
draw_samples(samples,frames, filename, filepath, labels_dataset, dataset,actions,joint_number,J);
%%%%%%%%%%%%%%%%%%%%%--view_2--%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% draw the view_2
frames = 10; % full length = 26
samples = 689;
filename = 'Train_Raw_cv3.h5';
labels_dataset = 'y_train';
dataset = 'x_train';
figure(2)
draw_samples(samples,frames, filename, filepath, labels_dataset, dataset,actions,joint_number,J);
%%%%%%%%%%%%%%%%%%%%%--view_3--%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% draw the view_3
frames = 10; % full length = 21
samples = 239;
actions = 10;
filename = 'Test_Raw_cv3.h5';
labels_dataset = 'y_test';
dataset = 'x_test';
%% to save the labels
% labels2 = h5read([filepath filename],['/' labels_dataset],[1,1],[actions,test_samples]);
% save labels2 labels2;
figure(3)
draw_samples(samples,frames, filename, filepath, labels_dataset, dataset,actions,joint_number,J);
end

function draw_samples(samples,frames, filename, filepath, labels_dataset, dataset,actions,joint_number,J)
h5disp([filepath filename],['/' dataset])
h5disp([filepath filename],['/' labels_dataset])
labels = h5read([filepath filename],['/' labels_dataset],[1,samples],[actions,1]);
data = h5read([filepath filename],['/' dataset],[1,1,samples],[joint_number*3,frames,1]);
skeleton = reshape(data',[frames,joint_number,3]);

% apply affine transformation on skeleton
% skeleton = affine_tranform(skeleton);

B(:,:,1) = skeleton(:,1:3:end);
B(:,:,2) = skeleton(:,2:3:end);
B(:,:,3) = skeleton(:,3:3:end);

X=B(:,:,1)';
Z=B(:,:,2)';
Y=B(:,:,3)';

for s=1:size(X,2)
    S=[X(:,s) Y(:,s) Z(:,s)];
    
    subplot(1,2,1);
    xlim = [0 10];
    ylim = [0 10];
    zlim = [0 10];
    set(gca, 'xlim', xlim, ...
        'ylim', ylim, ...
        'zlim', zlim);
    
    h=plot3(S(:,1),S(:,2),S(:,3),'r.','MarkerSize',20);
    %rotate(h,[0 45], -180);
    set(gca,'DataAspectRatio',[1 1 1])
    %     axis([-1 1 -1 1 -1 1])
    
    
    for j=1:joint_number-1
        c1=J(1,j);
        c2=J(2,j);
        line([S(c1,1) S(c2,1)], [S(c1,2) S(c2,2)], [S(c1,3) S(c2,3)],'LineWidth',2);
    end
    axis off;
    ssi = compute_ssm(skeleton(s,:));
%     ssi1 = h5read([filepath 'Trainset3.h5'],['/' 'X_train'],[1,1,1,s,samples],[1,joint_number,joint_number,1,1]);
%     ssi1 = squeeze(ssi1);
    subplot(1,2,2);imagesc(ssi);axis image;
    axis off;
    saveas(gcf, ['output3_',num2str(s)], 'bmp')
    pause(1/15)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1);
% S=[X(:,1) Y(:,1) Z(:,1)];
% joints=plot3(S(:,1),S(:,2),S(:,3),'rs','markersize',10);
% 
% for j=1:joint_number-1
%     c1=J(1,j);
%     c2=J(2,j);
%     plot3([S(c1,1) S(c2,1)], [S(c1,2) S(c2,2)], [S(c1,3) S(c2,3)],'-rs','LineWidth',2);hold on;
% end
% S=[X(:,size(X,2)) Y(:,size(X,2)) Z(:,size(X,2))];
% joints=plot3(S(:,1),S(:,2),S(:,3),'rs','markersize',10);hold on;
% for j=1:joint_number-1
%     c1=J(1,j);
%     c2=J(2,j);
%     plot3([S(c1,1) S(c2,1)], [S(c1,2) S(c2,2)], [S(c1,3) S(c2,3)],'-bs','LineWidth',2);hold on;
% end
% S_begin=[X(:,1) Y(:,1) Z(:,1)];
% S_end = [X(:,size(X,2)) Y(:,size(X,2)) Z(:,size(X,2))];
% 
% for j=1:size(X,2)-1
%     S=[X(:,j) Y(:,j) Z(:,j)];S_next = [X(:,j+1) Y(:,j+1) Z(:,j+1)];
%     for i = 1:20
%         axis equal;plot3([S(i,1) S_next(i,1)],[S(i,2) S_next(i,2)],[S(i,3) S_next(i,3)],'-k','LineWidth',1);hold on;
%     end
% end

% class_num = length(unique((trainGID)));
% for i = 1:class_num
%     trajectory = TRAJDB(2,(trainGID == i));
%     class_len = length(trajectory);
%     for j = 1:class_len
%     plot3d(trajectory{1,j});
%     xlabel('X','FontWeight','bold');ylabel('Y','FontWeight','bold');zlabel('Z','FontWeight','bold');
%     saveas(gcf,strcat(num2str(i),'-',num2str(j)),'png')
%     end
% end
end

function ssi = compute_ssm(skeleton)
    skeleton = reshape(skeleton,[3,20])';
    ssi = pdist2(skeleton,skeleton);
    % ssi = ssi/(max(max(ssi))+1e-8);
end

function skeleton = affine_tranform(skeleton)

for n = 1:size(skeleton,1)
    skeleton_time = reshape(skeleton(n,:),[3,20])';
    Sx = 1.2;
    Sy = 1.0;
    Sz = 1.8;
    tform = affine3d([Sx 0 0 0; 0 Sy 0 0; 0 0 Sz 0; 0 0 0 1]);
    num_joint = size(skeleton_time,1);
    for t = 1:num_joint
        skeleton_time(t,:) = transformPointsForward(tform,skeleton_time(t,:));
    end
    skeleton(n,:) = reshape(skeleton_time',[1,num_joint*3]);
end

end

