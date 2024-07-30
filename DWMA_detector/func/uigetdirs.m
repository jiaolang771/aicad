%
% return a cell array of subfolders of selected folder


function subdirs=uigetdirs(folder)


subdirs = [];
selec_folder = uigetdir(folder,'Select directory containing all subject folders with NIFTI files');

if selec_folder==0
    return;
end
tmp = dir(selec_folder);


for i=3:size(tmp,1)
  subdirs = [subdirs; {strcat(selec_folder,'\',tmp(i).name)}];
end
