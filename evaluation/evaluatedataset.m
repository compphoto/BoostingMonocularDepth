clc;
clear

%% 
evaluation_name = 'testEvaluation';

%% Fill in the needed path and flags for evaluation
estimation_path = '';
gt_depth_path = '';
evaluation_matfile_save_dir = './';
dataset_disp_gttype = true; %% (True) for gt disparity (False) for gt depth 
superpixel_scale = 1; % use 0.2 for middleburry and 1 for ibims1 and NYU %Used for rescaling the image before extracting superpixel centers for D3R error metric. smaller scale for high res images results in a faster evaluation.

%%

imglist = dir(fullfile(gt_depth_path,'*.png'));
fprintf('Estimation path: %s\nGT path: %s\nGT type:%d (0:depth 1:disparity)\nTotal number of images: %d \n',estimation_path,gt_depth_path,dataset_disp_gttype,numel(imglist))

for img=1:numel(imglist)
    imagename = imglist(img).name;
    if dataset_disp_gttype
        gt_disp = im2double(imread(fullfile(gt_depth_path,sprintf('%s',imagename))));
        gt_disp(gt_disp==0)=nan;
        min_gt_disp = min(gt_disp(:));
        gt_depth = 1./gt_disp;
        gt_disp = rescale(gt_disp,0,1);
        gt_depth = rescale(gt_depth,0,1);
    else
        gt_depth = im2double(imread(fullfile(gt_depth_path,sprintf('%s',imagename))));
        gt_depth(gt_depth==0)=nan;
        gt_disp = 1./gt_depth;
        gt_disp = gt_disp / max(gt_disp(:));
        gt_depth(gt_disp>1)=nan;
        gt_disp(gt_disp>1)=nan;
        min_gt_disp = min(gt_disp(:));
        gt_disp = rescale(gt_disp,0,1);
        gt_depth = rescale(gt_depth,0,1);
    end
        
    estimate_disp = im2double(imread(fullfile(estimation_path,sprintf('%s',imagename))));
    estimate_disp_ = rescale(estimate_disp,min_gt_disp,1);
    estimate_depth = 1./estimate_disp_;
    estimate_depth = rescale(estimate_depth,0,1);
    
    gt_small=imresize(gt_disp,superpixel_scale,'nearest');
    samples=5000;
    [centers,neightbouring_rel,random_rel]=extractD3Rpoints(gt_small,samples);
    [sub_x,sub_y] = ind2sub(size(gt_small),centers);
         
    sub_x = sub_x/superpixel_scale;
    sub_y = sub_y/superpixel_scale;
    centers = sub2ind(size(gt_depth),sub_x,sub_y);  
        
    hf_ratio = 0.1;
    d3r_ratio = 0.01;
     
    [confidence,error]=D3R(gt_depth, estimate_depth, centers, neightbouring_rel,hf_ratio,d3r_ratio);
    d3r_error(img)=error/confidence;
    
    hf_ratio = 0;
    d3r_ratio = 0.03;
    [confidence,error]=D3R(gt_depth, estimate_depth, centers, random_rel,hf_ratio,d3r_ratio); 
    ord_error(img)=error/confidence; 
    
    mask = ones(size(gt_depth));
    mask(isnan(gt_disp))=0;

    gt_disp(mask==0)=0;
    estimate_disp(mask==0)=0;
    rmse_error(img) = sqrt(immse(gt_disp,estimate_disp));
    
    mask(gt_depth==0)=0;
    mask(estimate_depth==0)=0;
    
    gt_depth = gt_depth(:);
    estimate_depth = estimate_depth(:);
    
    gt_depth(mask(:)==0) = [];
    estimate_depth(mask(:)==0) = [];
     
    thresh = max(gt_depth./estimate_depth, estimate_depth./gt_depth);
    thresh_1_25_error(img) = length(find(thresh>1.25)) / numel(thresh(:));
    thresh_1_25_2_error(img) = length(find(thresh>1.25^2)) / numel(thresh(:));
    thresh_1_25_3_error(img) = length(find(thresh>1.25^3)) / numel(thresh(:));
    log_10(img) = mean(abs(log10(gt_depth) - log10(estimate_depth)),'all');
    abs_rel(img) = mean(abs(gt_depth - estimate_depth)./ gt_depth,'all');
    sq_rel(img) = mean(((gt_depth - estimate_depth).^2)./ gt_depth,'all');
    
    fprintf('(%d/%d) - %s\n\t D3R:%0.4f RMSE:%0.4f ABSREL:%0.4f\n',img,numel(imglist),imagename,d3r_error(img),rmse_error(img),abs_rel(img))
end

save(fullfile(evaluation_matfile_save_dir,sprintf('evaluation_%s.mat',evaluation_name)),'ord_error','d3r_error','rmse_error','thresh_1_25_error','thresh_1_25_2_error','thresh_1_25_3_error','log_10','abs_rel','sq_rel');

%%
fprintf('ORD: %0.4f \n',mean(ord_error(:),'omitnan'))
fprintf('D3R: %0.4f \n',mean(d3r_error(:),'omitnan'))
fprintf('RMSE: %0.4f \n',mean(rmse_error(:),'omitnan'))
fprintf('THRESH 1.25: %0.4f \n',mean(thresh_1_25_error(:),'omitnan'))
fprintf('THRESH 1.25^2: %0.4f \n',mean(thresh_1_25_2_error(:),'omitnan'))
fprintf('THRESH 1.25^3: %0.4f \n',mean(thresh_1_25_3_error(:),'omitnan'))
fprintf('LOG10: %0.4f \n',mean(log_10(:),'omitnan'))
fprintf('ABS REL: %0.4f \n',mean(abs_rel(:),'omitnan'))
fprintf('SQ REL: %0.4f \n',mean(sq_rel(:),'omitnan'))
