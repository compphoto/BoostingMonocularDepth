clc
clear
%%

load('ibims1_selected.mat');

root_dir = '';

ibims1_all_files = dir(sprintf('%s/ibims1/rgb/*.png',root_dir)); 

for i=1:numel(ibims1_all_files)
    current_name = ibims1_all_files(i).name;
    if ismember(current_name,files)
        fprintf('%d - %s \n',i,current_name)
    else
        delete(fullfile(ibims1_all_files(i).folder,ibims1_all_files(i).name))
    end
end