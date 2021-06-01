function [centers,neightbouring_rel,random_rel]=extractD3Rpoints(gt,samples)
% Inputs
% gt: ground truth image to exctact the superpixels from
% samples: expected number of samples (actual number of samples might be slightly different)

% Outputs:
% centers: superpixel centers to be used within D3R function
% neightbouring_rel: neighbouring point pair relations to be used within D3R function
% random_rel:  randomly selected point pair relations to be used within D3R function

gt(isnan(gt))=0;
[L,NumLabels]=superpixels(gt,samples,'Compactness',20);

idx = label2idx(L);
[height, width, ~] = size(gt);
neightbouring_rel = cell(NumLabels,1);    
random_rel = cell(NumLabels,1);
centers = zeros(NumLabels,1);
for i = 1:NumLabels
    mask  = L == i;
    neightbouring_rel{i} = unique(L(bwdist(mask ,'euclidean')==1));
    random_rel{i} = randi([1,NumLabels],3,1);   
    a = idx{i};
    center = computeCenter(a,height);
    centers(i) = center;
end

end

function center = computeCenter(cluster,imgheight)
sumx=0;
sumy=0;
for i=1:length(cluster)
    [x,y] = index2index(cluster(i),imgheight);
    sumx = sumx + x;
    sumy = sumy + y;
end
center = floor(sumx/length(cluster)) + (floor(sumy/length(cluster))-1)*imgheight;
end