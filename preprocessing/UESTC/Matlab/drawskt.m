function drawskt()

    J = [24     12     11     10    9    21     4   3    8    8    7    6     5    21   2    1 ...
        17    18   19   20  13  14  15  16;
       12     25     12     11    10    9     3   21   22   23   8    7     6    5    21   2 ...
           1     17   18   19   1  13  14  15];

    load a30_d3_p118_c2_skeleton.mat;
    skeleton = v;
    B(:, :, 1) = skeleton(:, 1:3:end);
    B(:, :, 2) = skeleton(:, 2:3:end);
    B(:, :, 3) = skeleton(:, 3:3:end);

    X = B(:, :, 1)';
    Z = B(:, :, 2)';
    Y = B(:, :, 3)';

    %% plot animation of actions
    for s = 1:size(X, 2)
        S = [X(:, s) Y(:, s) Z(:, s)];

        xlim = [0 800];
        ylim = [0 800];
        zlim = [0 800];
        set(gca, 'xlim', xlim, ...
            'ylim', ylim, ...
            'zlim', zlim);

        h = plot3(S(:, 1), S(:, 2), S(:, 3), 'r.');
        % rotate(h,[0 45], -180);
        set(gca, 'DataAspectRatio', [1 1 1])
        %     axis([0 400 0 400 0 400])

        for j = 1:24
            c1 = J(1, j);
            c2 = J(2, j);
            line([S(c1, 1) S(c2, 1)], [S(c1, 2) S(c2, 2)], [S(c1, 3) S(c2, 3)]);
        end

        pause(1 / 20)
    end
