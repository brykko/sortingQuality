function [clusterIDs, unitQuality, contaminationRate] = maskedClusterQualitySparse(clu, fet, fetInds, fetNchans, useMex)
% - clu is 1 x nSpikes
% - fet is nSpikes x nPCsPerChan x nInclChans
% - fetInds is nClusters x nInclChans (sorted in descending order of
% relevance for this template)
% - fetN is an integer, the number of features to

if nargin < 4 || isempty(fetNchans), fetNchans = min(4, size(fetInds,2)); end % number of channels to use
if nargin < 5 || isempty(useMex), useMex = true; end

nFetPerChan = size(fet,2);
fetN = fetNchans*nFetPerChan; % now number of features total

N = numel(clu);
assert(fetNchans <= size(fet, 3) && size(fet, 1) == N , 'bad input(s)')

% Sort features by cluster index. This enables much more efficient
% extraction of spike features by cluster.
[clu, iSort] = sort(clu);
fet = fet(iSort, :, :);

clusterIDs = unique(clu);
unitQuality = zeros(size(clusterIDs));
contaminationRate = zeros(size(clusterIDs));

fprintf('%12s\tQuality\tContamination\n', 'ID'); % comment to suppress printing out the intermediate results
for c = 1:numel(clusterIDs)
    
    theseSp = clu==clusterIDs(c);
    n = sum(theseSp); % #spikes in this cluster
    if n < fetN || n >= N/2
        % cannot compute mahalanobis distance if less data points than
        % dimensions or if > 50% of all spikes are in this cluster
        unitQuality(c) = 0;
        contaminationRate(c) = NaN;
        continue
    end
    
    fetThisCluster = reshape(fet(theseSp,:,1:fetNchans), n, []);
    
    % now we need to find other spikes that exist on the same channels
    theseChans = int16(fetInds(c,1:fetNchans));
    
    % for each other cluster, determine whether it has at least one of
    % those channels. If so, add its spikes, with its features put into the
    % correct places
    if useMex
        try
            [fetOtherClusters, nSpikes] = maskedClusterQualitySparseMex( ...
                fetNchans, fet, clu, theseChans, clusterIDs, c-1, int16(fetInds));
            fetOtherClusters = fetOtherClusters(1:nSpikes, :, :);
        catch e
            fprintf('Failed to call maskedClusterQualitySparseMex: "%s". Reverting to basic implementation\n', e.message);
            fetOtherClusters = getOtherFeatures(fet, fetNchans, clu, clusterIDs, c, fetInds, theseChans);
        end
    else
        % Slightly faster version of the original code (below, commented)
        fetOtherClusters = getOtherFeatures(fet, fetNchans, clu, clusterIDs, c, fetInds, theseChans);
    end
    
    % ORIGINAL VERSION
    %         tic();
    %         nInd = 1; fetOtherClusters2 = zeros(0,size(fet,2),fetNchans); theseOtherSpikes = 1;
    %         for c2 = 1:numel(clusterIDs)
    %             if c2~=c
    %                 idx = find(theseOtherSpikes, 1, 'last')+1;
    %                 fprintf('Starting query clu%u at spike idx %u\n', c2, idx);
    %                 chansC2Has = fetInds(c2,:);
    %                 for f = 1:length(theseChans)
    %
    %                     fprintf('ismember(%u, [%u, %u, %u, %u])\n', theseChans(f), chansC2Has(1), chansC2Has(2), chansC2Has(3), chansC2Has(4))
    %                     if ismember(theseChans(f), chansC2Has)
    %                         theseOtherSpikes = clu==clusterIDs(c2);
    %                         thisCfetInd = find(chansC2Has==theseChans(f),1);
    %                         fetOtherClusters2(nInd:nInd+sum(theseOtherSpikes)-1,:,f) = ...
    %                             fet(theseOtherSpikes,:,thisCfetInd);
    %                         indsTmp = find(theseOtherSpikes);
    %                         fprintf('Appended clu%u chan%u spikes %u-%u fet idx %u, total spikes %u\n', ...
    %                             c2, theseChans(f), indsTmp(1), indsTmp(end), thisCfetInd, nInd+sum(theseOtherSpikes)-1);
    %                         tmp = fet(theseOtherSpikes,:,thisCfetInd);
    %                         tmp = tmp(1, :);
    %                         indsTmp = sub2ind(size(fet), find(theseOtherSpikes, 1), 1:3, f);
    %                         fprintf('First spike fet %u, fet_idx %u: ch1=%.3f, ch2=%.3f, ch3=%.3f\n', f, thisCfetInd, tmp(1), tmp(2), tmp(3));
    %                     end
    %
    %                 end
    %                 if any(ismember(chansC2Has, theseChans))
    %                     nInd = nInd+sum(theseOtherSpikes);
    %                 end
    %             end
    %         end
    %         tOrig = toc();
    
    fetOtherClusters = reshape(fetOtherClusters, size(fetOtherClusters,1), []);
    
    [uQ, cR] = maskedClusterQualityCore(fetThisCluster, fetOtherClusters);
    
    unitQuality(c) = uQ;
    contaminationRate(c) = cR;
    
    fprintf('cluster %3d: \t%6.1f\t%6.2f\n', clusterIDs(c), unitQuality(c), contaminationRate(c)); % comment to suppress printing out the intermediate results
    
    if uQ>1000
        keyboard;
    end
    
end

end

function fetOther = getOtherFeatures(fet, fetNchans, clu, clusterIDs, c, fetInds, theseChans)
% Slightly faster version of the original (below, commented)
nInd = 1;
fetOther = zeros(size(fet, 1), size(fet,2), fetNchans);
theseOtherSpikes = 1;
for c2 = 1:numel(clusterIDs)
    if c2~=c
        chansC2Has = fetInds(c2,:);
        for f = 1:length(theseChans)
            if any(theseChans(f)==chansC2Has)
                theseOtherSpikes = clu==clusterIDs(c2);
                thisCfetInd = find(chansC2Has==theseChans(f),1);
                fetOther(nInd:nInd+sum(theseOtherSpikes)-1,:,f) = ...
                    fet(theseOtherSpikes,:,thisCfetInd);
            end
            
        end
        v = chansC2Has == theseChans';
        if any(v(:))
            nInd = nInd+sum(theseOtherSpikes);
        end
    end
end
fetOther = fetOther(1:nInd-1, :, :);
end