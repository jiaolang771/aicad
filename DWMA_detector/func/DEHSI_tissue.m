
%
%   DEHSI detection
%
function DEHSI_tissue

disp('=====================');
disp('Running...');
addpath(genpath([pwd '\func\']));

%%  select target subject;s folder
subdirs = uigetdir(pwd,'Select folder with segmented brain tissue maps');
if subdirs==0
    return
end

dtype = {'T2'};
dext = {'.nii','.img'};
flist = struct;
nsubs = 1;
sub = 0;
% for n=1:nsubs
n=1;
s = strsplit(subdirs,filesep);
files = cellstr(strtrim(ls(subdirs))); %strsplit(cellstr(strtrim(ls(subdirs{n}))));
all_files = cellstr(strtrim(ls(subdirs)));
% Check for nifti or analyze files
for ext_i=1:2
    ext_check = ~cellfun(@isempty,regexp(files,dext{ext_i}));
    if sum(ext_check)>=1 && ext_i==1
        ext_index = ext_i;
        files(cellfun(@isempty,regexp(files,dext{ext_i})))=[];   % Only consider NIFTI files
        files(~cellfun(@isempty,regexp(files,'.gz')))=[];   % Remove compressed files - SPM can't handle
    elseif sum(ext_check)>=1 && ext_i==2
        %    consider .img and .hdr
        ext_index = ext_i;
        files(cellfun(@isempty,regexp(files,dext{ext_index})))=[];
        % consider original T2 image
        
    end
end

if cellfun(@isempty,regexp(files,'^c1'))
    warndlg('   Segmented MRI T2 images not found.','Preprocessing Aborted');
    return
end

files(~cellfun(@isempty,regexp(files,'^bs')))=[];
files(~cellfun(@isempty,regexp(files,'^c1')))=[];
files(~cellfun(@isempty,regexp(files,'^c2')))=[];
files(~cellfun(@isempty,regexp(files,'^c3')))=[];
files(~cellfun(@isempty,regexp(files,'^m')))=[];
files(~cellfun(@isempty,regexp(files,'^wc1')))=[];
files(~cellfun(@isempty,regexp(files,'^wc2')))=[];
files(~cellfun(@isempty,regexp(files,'^wc3')))=[];
files(~cellfun(@isempty,regexp(files,'^wm')))=[];


% Check for structural file
if ~isempty(dtype)
    check = ~cellfun(@isempty,regexp(files,dtype{1}));
else
    return
end

if sum(check)==0
    h=warndlg('Error: Brain MRI T2 images not found.','Preprocessing Aborted');
    uiwait(h);
    return
elseif sum(check)>1
    temp = files(check);
    [sel,~]=listdlg('PromptString',['Multiple ' dtype{1} ' files found.  ' ...
        'Select file for preprocessing'],...
        'SelectionMode','single','ListSize',[478 200],'ListString',temp);
    if isempty(sel)
        h = warndlg({'No structural file selected.';'';...
            ['Preprocessing Aborted']});
        uiwait(h);
        return
    end
else
    sel=1;
end


flist(n).subid = s{end};
flist(n).dir = subdirs;
flist(n).(dtype{1}) = files{sel};

if cellfun(@isempty,regexp(all_files,['m' files{sel}]))
    warndlg('Error: Segmented MRI T2 images not found for selected file.','Preprocessing Aborted');
    return
end


%end
if isempty(fieldnames(flist))
    return
end


%%
%-----------------------------------------%
% Read processed files and calculate tissue volumes %
%-----------------------------------------%

addpath(genpath([pwd '\func\']));
%%   abmorality detection start

for x=1:length(flist)
    if isempty(flist(x).(dtype{1}))
        break %continue    % if no structural file exists for subject x, skip subject
    end
    [apath, aname, ext] = fileparts(fullfile(char(flist(x).dir),char(flist(x).(dtype{1}))));
    
    
    %% read WM C2 and GM C1 files, generate WM+GM C1+C2   job 1
    
    %  read .nii or .img c1bs_subID_T2/ C2bs_subID_T2
    %%V = niftiread(filename)   2017b matlab
    %  use nifti reader package in tools folder
    
    Brain_struct = load_untouch_nii(fullfile(apath,['mbs_' aname ext]));
    Brain = Brain_struct.img;  %  brain image
    
    GM_struct = load_untouch_nii(fullfile(apath,['c1bs_' aname ext]));
    GM = GM_struct.img;                                   %  GM probability map
    
    WM_struct = load_untouch_nii(fullfile(apath,['c2bs_' aname ext]));
    WM = WM_struct.img;                                   %  WM probability map
    
    CSF_struct = load_untouch_nii(fullfile(apath,['c3bs_' aname ext]));
    CSF = CSF_struct.img; 
    
    
    %%   apply probability map
    
    WM_pm_all = (WM>GM)&(WM>CSF)&(WM>0);
    GM_pm_all = (GM>WM)&(GM>CSF)&(GM>0);
    CSF_pm_all = (CSF>WM)&(CSF>GM)&(CSF>0);
    
%     disp(sum(CSF_pm_bin(:)));
%     disp(sum(CSF_pm_all(:)));

    
end

%%   display
disp('=====================');
disp('Done');

voxel_size = Brain_struct.hdr.dime.pixdim(2:4);
voxel_volume = voxel_size(1)*voxel_size(2)*voxel_size(3);

sum_volume_WM  = voxel_volume*sum(WM_pm_all(:)); 
sum_volume_GM  = voxel_volume*sum(GM_pm_all(:));
sum_volume_CSF  = voxel_volume*sum(CSF_pm_all(:));

f = msgbox({'Whole Brain Tissue Volume Calculation Completed';
  
    ['Whole brain white matter volume: ' num2str(round(sum_volume_WM,2))];
    ['Whole brain grey matter volume: ' num2str(round(sum_volume_GM,2))];
    ['Whole brain CSF volume: ' num2str(round(sum_volume_CSF,2))];
    });


%
%    end of code
%