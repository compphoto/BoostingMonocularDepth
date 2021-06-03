function generatecrops(dataset, blsize, stride, subsetname,counter_offset)

root_dir = '';

imgdir = sprintf("%s/%s/rgb",root_dir,dataset); 
est_lq_dir = sprintf("%s/%s/whole_low_est",root_dir,dataset);
est_hq_dir = sprintf("%s/%s/whole_high_est",root_dir,dataset);

img_result_dir = sprintf("%s/mergenetdataset/%s/rgb",root_dir,subsetname);
outer_result_dir = sprintf("%s/mergenetdataset/%s/outer",root_dir,subsetname);
gt_fake_result_dir = sprintf("%s/mergenetdataset/%s/gtfake",root_dir,subsetname);

mkdir(img_result_dir);
mkdir(outer_result_dir);
mkdir(gt_fake_result_dir);


imglist = dir(sprintf('%s/*.png',imgdir));

if strcmp(dataset,'middleburry')
    test_set_size = 2;
else
    test_set_size = 0;
end
if strcmp(subsetname,'train')
    i_start = 0;
    i_end = -test_set_size;
else
    i_start = numel(imglist)-test_set_size;
    i_end = 0;
end

for i=1+i_start:numel(imglist)+i_end
    samplename = erase(imglist(i).name,'.png');

    img = im2uint16(imread(sprintf('%s/%s.png',imgdir,samplename)));
    
    est_lq = im2uint16(imread(sprintf('%s/%s.png',est_lq_dir,samplename)));
    est_hq = im2uint16(imread(sprintf('%s/%s.png',est_hq_dir,samplename)));
    
    counter1 = counter_offset;
    for k = blsize:stride:size(img,2)-blsize
        counter2 = counter_offset;
        for j = blsize:stride:size(img,1)-blsize

            cropbounds = [j-blsize+1,k-blsize+1,j-blsize+2*blsize,k-blsize+2*blsize];
            sample_img = img(cropbounds(1):cropbounds(3),cropbounds(2):cropbounds(4),:);
            sample_hq_est = est_hq(cropbounds(1):cropbounds(3),cropbounds(2):cropbounds(4),:);
            sample_lq_est = est_lq(cropbounds(1):cropbounds(3),cropbounds(2):cropbounds(4),:);
       
            
            if size(sample_hq_est) == [2*blsize 2*blsize]
                
                sample_img = imresize(sample_img,[672,672]);
                sample_hq_est = imresize(sample_hq_est,[672,672]);
                sample_lq_est = imresize(sample_lq_est,[672,672]);

                imwrite(sample_img,sprintf('%s/%s_&%d_%d&.png',img_result_dir,samplename,counter1,counter2)) 
                imwrite(sample_hq_est,sprintf('%s/%s_&%d_%d&.png',gt_fake_result_dir,samplename,counter1,counter2)) 
                imwrite(sample_lq_est,sprintf('%s/%s_&%d_%d&.png',outer_result_dir,samplename,counter1,counter2))
                fprintf('(%d/%d) %s - [%d-%d] \n',i,numel(imglist),samplename,counter1,counter2);
            else
                fprintf('(%d/%d) %s - Size Issue: Not cropped! \n',i,numel(imglist),samplename)    
            end
            counter2 = counter2 + 1;
        end
        counter1 = counter1 + 1;
    end
       
end

end


