function [w_scale_a, w_shape_b] = weibull_params(hist)

    hist = hist(:);
    
    ind = (1:size(hist, 1));
    
    ind = reshape(ind, size(hist));
    
    %wtimer = tic;
    %[ph, pc] = wblfit(ind, [], [], hist);
	[ph, pc] = wblfit(hist, 5.0);
    %fprintf('  > Weibull: ');
    %toc(wtimer)
    
    w_shape_b = ph(2);
    w_scale_a = ph(1);
    
    return