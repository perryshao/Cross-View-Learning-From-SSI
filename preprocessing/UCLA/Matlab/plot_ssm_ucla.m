function plot_ssm_ucla(filename, dataset, samples, time, persons)
    joint_num = 20;
    dim_ssm = persons * joint_num;
    ssi = zeros(201, dim_ssm, dim_ssm);
    filepath = 'C:\Users\perry\Desktop\ntu3daction\';
    % belta = 1e-6;
    h5disp([filepath filename], ['/' dataset])
    for t = 1:201
        data = h5read([filepath filename], ['/' dataset], [1, 1, 1, t, samples], [1, dim_ssm, ...
            dim_ssm, 1, 1]);
        data_img = squeeze(data);
        %     data_img_k = 1-exp(-belta*data_img);
        ssi(t, :, :) = data_img;
    end
    save ssi ssi
    % data_img_k = tanh(belta*data_img);
    % sub1 = subplot(1,2,1);imagesc(data_img_k);
    figure(1)
    for t = 1:time
        sub1 = subplot(1, 2, 1); imagesc(squeeze(ssi(t, :, :))); axis image;
        % sub2 = subplot(1,2,2);imagesc(squeeze(ssi(time+1,:,:)));axis image;
        pause(1 / 20);
    end

    % Specify some parameters for the plot
    x0   = 1;  % spacing between and around figures (inches)
    y0   = 0;    % offset from the bottom (inches)
    w    = 2.25; % size of each subfigure (w x w inches)

    % Now specify each figure's location and dimensions
    % as [x  y  width  height]:
    % 'x' and 'y' are position of the lower-left corner of each panel
    % 'length' and 'height' are the dimensions of each panel
    set(sub1, 'Units', 'inches', 'Position', [x0     y0 + w + x0  w w]);
    set(sub2, 'Units', 'inches', 'Position', [w + 2 * x0 y0 + w + x0  w w]);
