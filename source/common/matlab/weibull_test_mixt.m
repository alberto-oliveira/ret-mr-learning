function [t] = weibull_test_unique_mixt(rankfpath, f, z, delta, r, nrm)
    % Computes a threshold on the score distribution of a certain query, using
    % the values of the tail of the score distribution. As a first step, the
    % distribution is converted to a distribution*, containing only unique
    % values of scores.
    % @param rankfpath: path of the file with the score distribution of a certain
    %                   query;
    % @param f: Value between 0.0 and 1.0 specifying the tail starting point. 
	%           Computed as (hist_u)*f
    % @param z: Value between 0.0 and 1.0 specifying the tail size. 
	%           Computed as (full_tail_size)*s
    % @param delta: confidence value passed to the inverse of the cumulative
    %               distribution function of the weibull distribution computed,
    %               used to retrieve the output threshold;
    % @param draw: flag indicating if output images are desired;
    %
    % @ouput t: threshold on the score score distribution. Any image with
    %           score value v > t will be considered relevant;
    
	rng_seed = 942010;
	
    rng(rng_seed);
    
    % -- PARAMETERS (AKA CONSTANTES DE ACOCHAMBRAMENTO)
    i = 1.0; % Iterator. Used to find the mean of thresholds as well
    max_iter = 30;
    min_diff = 2; % minimum difference to consider there was change in # of matches

    % Minimum size of the tail distribution
    min_tail_size = 5;
    
    % Tail Factor - from the starting point, Percent of elements to
    % be considered tail
    tail_f = 0.90;
    
    % Increase factor - Maximum percentual increase of the tail on
    % both ends
    inc_f = 0.20;
    
    % Decrease factor - Maximum percentual decrese of the tail on
    % both ends
    dec_f = 0.20;
    % -----------------------------------
    
    
    % Opens and reads the score distribution file
    % rankfpath
    fid = fopen(rankfpath);
    rank = textscan(fid, '%s %f %f %f %f');
    fclose(fid);
    
    % Score distribution
	hist = rank{2}(1:end)';
    
    % Based on the original score distribution, computes
    % a distribution* with no repeating values
    hist_u = fliplr(unique(hist));
    hsz = numel(hist_u); % size of distribution*
    
    % Top r positions we are interested in predicting. 
	% Excludes the first element.
    ridx = find(hist_u == hist(r), 1);
	tsz = hsz - ridx;
	
	%w = ceil(tail_f * tsz)
	%w = 25
    
    % Tail has size = hsz - ceil(hsz*f), where f is a factor
    % such that 0 <= f <= 1.0 (though 0.0 and 1.0 do not
    % make sense).
    midx = ceil(numel(hist_u)*f);
	full_tail_size = numel(hist_u) - midx;
	w = floor(full_tail_size*z);
	
    m = find(hist == hist_u(midx), 1);
    sidx = midx;
    eidx = midx + w;
	if eidx > hsz
	    eidx = hsz;
	end
    
    s = find(hist == hist_u(sidx), 1);
    
    orig_tail_size = eidx - sidx + 1;
    while hsz - ridx >= min_tail_size && orig_tail_size < min_tail_size && (sidx > ridx + 1 || eidx < hsz)
        if sidx - 1 >= ridx + 1
            sidx = sidx - 1;
        end
        
        if eidx + 1 <= hsz
            eidx = eidx + 1;
        end
		orig_tail_size = eidx - sidx + 1;
    end
    
    %fprintf('Distribution size: %d\n', numel(hist_u));
    %fprintf('Distribution size - top r: %d\n\n', numel(hist_u) - ridx);

    
    % Random perturbations
    ms_bot = ceil(orig_tail_size*inc_f); % Max beginning perturbation bottom
    ms_top = ceil(orig_tail_size*dec_f); % Max beginning perturbation top
    me_bot = ceil(orig_tail_size*dec_f); % Max end perturbation bottom
    me_top = ceil(orig_tail_size*inc_f); % Max end perturbation top
	
	%fprintf('Start: -%d to %d\n', ms_bot, ms_top);
	%fprintf('Top: -%d to %d\n', me_bot, me_top);
    
    vec_ms = (-ms_bot:ms_top);
    vec_me = (-me_bot:me_top);
    
    max_comb = max(size(combvec(vec_ms, vec_me)));
    
    % Initial value of tail start and end
    n_sidx = sidx;
    n_eidx = eidx;

    wbl_dist_vec = [];
    tail_size_vec = [];
    
    used_pairs = [0, 0];

    while i <= max_iter && i <= max_comb
        
        % Generates tail. The first iteration has the
        % input tail parameters. Subsequent iterations are
        % perturbations on beginning and end of the initial
        % tail. The starting index must start after the top
        % r+1 element.
        if i > 1
            n_sidx = 0;
            n_eidx = 0;
            spert = 0;
            epert = 0;
            while n_eidx - n_sidx < min_tail_size && ismember([spert, epert], used_pairs, 'rows') ~= 0
                spert = randi([-ms_bot ms_top]);
                epert = randi([-me_bot me_top]);
                
                n_sidx = sidx + spert;
                n_eidx = eidx + epert;       
            end
            
            if n_sidx < ridx + 1
                n_sidx = ridx + 1;
            end
            
            if n_eidx > hsz
                n_eidx = hsz;
            end
            
        end
        
        used_pairs = [used_pairs; [n_sidx, n_eidx]];
        
        % Gets the tail of the score distribution, used to
        % estimate weibull. The tail is computed starting
        % from
        % hist_tail = hist_u(sidx:n);
        hist_tail = hist_u(n_sidx:n_eidx);
        tail_size = numel(hist_tail);
        %fprintf('Tail from %d ~ %d  --  size = %d\n', n_sidx, n_eidx, tail_size);
        
        tail_size_vec = [tail_size_vec tail_size];
    
        % Computes the Scale and Shape parameters of weibull
        % using matlab's wblfit function, which employs MLE
        [wa, wb] = weibull_params(hist_tail);      
    
        % If either the scale or the shape parameter is
        % infinite, there was some issue with computing
        % the weibull for the given tail, thus sets the
        % threshold to inf
        if wb ~= Inf && wa ~= Inf
            wbl_dist_vec = [wbl_dist_vec; [wa, wb]];
            %fprintf('scale = %0.2f | shape = %0.2f\n\n', wa, wb);
        end
        
        i = i + 1;
        
    end
    
	
	% -- STARTS WEIBULL MIXING --
    wbl_samples = [];
    
    % Need to tranpose because of the way the matrix is organized
    for row=wbl_dist_vec'
        wa_i = row(1);
        wb_i = row(2);
		
		if ~isnan(wa_i) && ~isnan(wb_i)
		    %fprintf('%f - %f\n', wa_i, wb_i);
			wbl_samples = [wbl_samples wblrnd(wa_i, wb_i, 1, 50)];
		end
    
    end
    
	if ~isempty(wbl_samples)
		[ph, pc] = wblfit(wbl_samples);
    
		w_b_mix = ph(2);
		w_a_mix = ph(1);
		
		t = wblinv(delta, w_a_mix, w_b_mix);
	else
	    t = Inf;
	end
	
	return