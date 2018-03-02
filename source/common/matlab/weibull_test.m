function [t] = weibull_test_unique(rankfpath, f, z, delta)
    % Computes a threshold on the score distribution of a certain query, using
    % the values of the tail of the score distribution. As a first step, the
    % distribution is converted to a distribution*, containing only unique
    % values of scores.
    % @param rankfpath: path of the file with the score distribution of a certain
    %                   query;
    % @param f: 
    % @param z: 
    % @param delta: confidence value passed to the inverse of the cumulative
    %               distribution function of the weibull distribution computed,
    %               used to retrieve the output threshold;
    % @param draw: flag indicating if output images are desired;
    %
    % @ouput t: threshold on the score score distribution. Any image with
    %           score value v > t will be considered relevant;
    
    o_n = z;
    % Opens and reads the score distribution file
    fid = fopen(rankfpath);
    rank = textscan(fid, '%s %f %f %f %f');
    fclose(fid);
	
    % Score distribution
	hist = rank{2}(1:end)';

    
    numel(hist);
    
    % Based on the original score distribution, computes
    % a distribution* with no repeating values
    hist_u = fliplr(unique(hist));
	hsz = numel(hist_u);
    
    % Positons of interest (for drawing)
    k = 10;
    k_u = find(hist_u == hist(k), 1);
	
	%fprintf('Setting s as median\n');
    
    % sidx is the position in the unique score distribution
    % whose value is the same as the value of position s
    % in the original score distribution
	
    sidx = ceil(hsz*f);
	full_tail_size = hsz - sidx;
	w = floor(full_tail_size*z);
	
	s = find(hist == hist_u(sidx), 1);
	eidx = sidx + w;
    if eidx > hsz
        eidx = hsz;
    end
    e = find(hist == hist_u(eidx), 1);
	
    %fprintf('initial position: %d - Tail size: %d\n', sidx, w+1);
    
    % Gets the tail of the score distribution, used to
    % estimate weibull. The tail is computed starting
    % from
    hist_tail = hist_u(sidx:eidx);
	%hist_tail = hist_u(sidx:end);   % FULL TAIL
    
    % Computes the Scale and Shape parameters of weibull
    % using matlab's wblfit function, which employs MLE
	hist_tail(:)'
    [wa, wb] = weibull_params(hist_tail);
    
    % If either the scale or the shape parameter is
    % infinite, there was some issue with computing
    % the weibull for the given tail, thus sets the
    % threshold to inf
    if wb == Inf || wa == Inf
        t = Inf;
    else
        t = wblinv(delta, wa, wb);
    end
	
	return