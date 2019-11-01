% DPS Correntropy based, hierarchical density preserving data split
% 
%   R	    = DPS(A,LEVELS,LABS)
%   [R H]   = DPS(A,LEVELS,LABS)
%
% INPUT
%   A			Input data (rows = observations)
%   LEVELS		Number of split levels, default: 3
%   LABS        Labels for the data (optional, if no labels are given unsupervised split is performed)
%
% OUTPUT
%   R			Index array with rotation set with 2^LEVELS folds
%   H			Hierarchy of splits
%
% DESCRIPTION
% Density Preserving Sampling (DPS) divides the input dataset into a given
% number of folds (2^LEVELS) by maximizing the correntropy between the folds
% and can be used as an alternative for cross-validation. The procedure is
% deterministic, so unlike cross-validation it does not need to be repeated.
%
% REFERENCE
%   Budka, M. and Gabrys, B., 2012.
%   Density Preserving Sampling: Robust and Efficient Alternative to Cross-validation for Error Estimation.
%   IEEE Transactions on Neural Networks and Learning Systems, DOI: 10.1109/TNNLS.2012.2222925. 
  
function [R,H] = dps(A,levels,labs)
	
 	if (nargin<3) || isempty(labs), labs = ones(size(A,1),1); end
 	if (nargin<2) || isempty(levels), levels = 3; end
	
	% renumber the labels
	if min(labs)<1, labs = labs + 1 + abs(min(labs)); end
	u = unique(labs);
	t = nan(max(u),1);
	t(u) = 1:numel(u);
	labs = t(labs);
	
	H = zeros(levels,size(A,1));

	idxs = cell(1,levels+1);
	idxs{1} = {1:size(A,1)};
	for i = 1:levels
		for j = 1:2^(i-1)
			t = helper(A(idxs{i}{j},:),labs(idxs{i}{j},:));
			idxs{i+1}{2*j-1} = idxs{i}{j}(t{1});
			idxs{i+1}{2*j} = idxs{i}{j}(t{2});
		end
		for j = 1:length(idxs{i+1})
			H(i,idxs{i+1}{j}) = j;
		end
	end
	
	R = H(end,:);

end

function idxs = helper(A,labs)

	% calculate class sizes and indexes
	c = max(labs);
	cidx = cell(1,c);
	for i = 1:c, cidx{i} = find(labs==i); end
	csiz = cellfun(@(x) numel(x),cidx);
	
	% if the classes aro too small to be divided further, switch to classless mode
	% if any(csiz < 1), c = 1;	end  % BUG HERE
	 
	siz = zeros(2,1);
	idx = cell(1,c);

	for i = 1:c
		BI = cidx{i};
		B = A(BI,:);
		
		% mask is used for counting remaining objects
		m = length(BI);
		mask = true(1,m);		
		
		% working distance matrix
		BB = (B.*B)*ones(fliplr(size(B)));
		D = BB - 2*(B*B') + BB';
		D = D + diag(inf(m,1));

		Dorg = D;											% original distance matrix
		idx{i} = nan(2,ceil(m/2));
		
		for j = 1:floor(m/2)
			[mD,I] = min(D,[],1);							% \
			[~,J] = min(mD);								%   find two closest objects
			I = I(J(1)); J = J(1);							% /

			mask(I) = 0; mask(J) = 0;						% mark them as used
			
			% split the objects to maximally increase coverage of both subsets
			if (mean(Dorg(I,idx{i}(1,1:j-1))) + mean(Dorg(J,idx{i}(2,1:j-1))) < ...
				mean(Dorg(I,idx{i}(2,1:j-1))) + mean(Dorg(J,idx{i}(1,1:j-1))))
				idx{i}(1,j) = J;
				idx{i}(2,j) = I;
			else  
				idx{i}(1,j) = I;
				idx{i}(2,j) = J;
			end
			
			% remove used objects from the distance matrix
			D(I,:) = inf; D(:,I) = inf;
			D(J,:) = inf; D(:,J) = inf;
		end
		if isempty(j), j = 0; end							% in case the loop is not entered at all
		
		% odd number of points in class
		if sum(mask)>0
			I = find(mask);
			if siz(1)<siz(2)
				idx{i}(1,end) = I;
			elseif siz(1)>siz(2)
				idx{i}(2,end) = I;
			else
				if (mean(Dorg(I,idx{i}(1,1:j))) < mean(Dorg(I,idx{i}(2,1:j))))
					idx{i}(2,j+1) = I;
				else
					idx{i}(1,j+1) = I;
				end
			end
		end
		
		% convert indexes from class-specific to dataset-specific
		idx{i}(1,~isnan(idx{i}(1,:))) = BI(idx{i}(1,~isnan(idx{i}(1,:))));
		idx{i}(2,~isnan(idx{i}(2,:))) = BI(idx{i}(2,~isnan(idx{i}(2,:))));

		% update fold sizes
		siz(1) = siz(1) + sum(~isnan(idx{i}(1,:)));
		siz(2) = siz(2) + sum(~isnan(idx{i}(2,:)));

	end

	idx = cell2mat(idx);
	
	idxs = cell(1,2);
	idxs{1} = idx(1,~isnan(idx(1,:)));
	idxs{2} = idx(2,~isnan(idx(2,:)));
	
end
