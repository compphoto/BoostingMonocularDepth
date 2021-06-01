function [confidence,error] = D3R(gt,depth_est,center_points,point_pairs,freq_ratio,d3r_ratio)
% Inputs
% gt: ground truth.
% depth_est: estimation.
% center_points: superpixel center location from extractD3Rpoints function.
% point_pairs: point pair relations from extractD3Rpoints function.
% freq_ratio: ratio used to extract high frequnecies depth changes ... setting to zero means all frequencies must be taken into account.
% d3r_ratio: metric ratio.

% Outputs
% error: some of the errors
% confidence: number of the point pairs used to compute the error. (error value should be normalized according to the confidence to be comparable)


gt(isnan(gt))=0;
inflamed_gt =gt;% imadjust(gt);
same_ratio_gt = 1+d3r_ratio;
same_ratio_est = 1+d3r_ratio;
error = 0;
confidence = 0;
for i=1:length(center_points)
    neighbours = point_pairs{i};
    if isempty(neighbours)
        continue;
    end
    for j = neighbours'
        j_neighbours = point_pairs{j};
        j_neighbours(j_neighbours == i) = [];
        if isempty(j_neighbours)
             point_pairs{j} = [];
        else
             point_pairs{j} = j_neighbours;
        end
        index1=center_points(i);
        index2=center_points(j); 
        if(gt(index1)~=0 && gt(index2)~=0) % error in GT
             if ord(gt(index1),gt(index2),1+freq_ratio)~=0
                er = abs(ord(inflamed_gt(index1),inflamed_gt(index2),same_ratio_gt) - ord(depth_est(index1),depth_est(index2),same_ratio_est));
                error = error + er;
                confidence = confidence + 1;
             end
        end
    end
end

end
