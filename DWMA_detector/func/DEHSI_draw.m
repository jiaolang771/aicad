
%
%   DEHSI detection
%
function DEHSI_draw

disp('=====================');
disp('Running...');

addpath(genpath([pwd '\func\']));
%%  select target subject;s folder
subdirs = uigetdir(pwd,'Select folder with detected brain DWMA maps');
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
% Read processed files and show DWMA %
%-----------------------------------------%

addpath(genpath([pwd '\func\']));
%%   abmorality detection start



[apath, aname, ext] = fileparts(fullfile(char(flist(1).dir),char(flist(1).(dtype{1}))));
    

%%  input parameters
    dpara = inputdlg([{'Intensity cutoff of interest'},{'Interested slice'}],...
        'Enter parameter for DWMA manual correction',repmat([1 50],1,1),...
        [{'1.4'},{'70'}]); 
    
    if isempty(dpara)
       return 
    end
    
    std_coff = str2double(dpara{1});
    slice_b= str2double(dpara{2})+1;
    slice_a = slice_b;

%%  load data

if exist(fullfile(apath,[aname '_BRAIN.mat']), 'file')==2
    load(fullfile(apath,[aname '_BRAIN.mat']));
else   
    h=warndlg('Error: Brain MRI data not found.','Processing Aborted');
    uiwait(h);
    return
end

if exist(fullfile(apath,[aname '_' num2str(std_coff) '_DWMA_mask.mat']), 'file')==2  
    load(fullfile(apath,[aname '_' num2str(std_coff) '_DWMA_mask.mat']));
else  
    h=warndlg('Error: DWMA mask for interested cutoff not found.','Processing Aborted');
    uiwait(h);
    return
end

   %%  manually draw a contour for a subject
   
%brain_slice = Brain(:,:,slice_b);
DWMA_slice = Combo_2D_abnorm(:,:,slice_b);
tmp_Combo_2D_abnorm = Combo_2D_abnorm;

                       
%%   save, dispose or continue
indx=2;

while indx~=1
    close all;
    dwma_display_color(Brain,tmp_Combo_2D_abnorm,slice_b,slice_b);
    h = imfreehand;
    if isempty(h)
       break 
    end
    manual_mask = createMask(h);
    correction = rot90(manual_mask, -1);
    DWMA_slice = DWMA_slice.*(~correction);
    tmp_Combo_2D_abnorm(:,:,slice_b)= DWMA_slice;
    close all
    dwma_display_color(Brain,tmp_Combo_2D_abnorm,slice_b,slice_b);
    %pause(2);
    list = {'Save & Exit','Continue'};
    [indx,tf] = listdlg('ListString',list,'PromptString','Select a operation:',...
        'SelectionMode','single',...
        'ListSize',[150,40]);
    if tf==0
        warndlg('Operation canceled by user! Data are not saved','Processing Aborted');
        break       
    end
end

if indx==1
    Combo_2D_abnorm = tmp_Combo_2D_abnorm;
    save(fullfile(apath,[aname '_' num2str(std_coff) '_DWMA_mask.mat']),'Combo_2D_abnorm');
end



%%   calculate DWMA volume
if ~isempty(h) && tf~=0
    
    voxel_volume = voxel_size(1)*voxel_size(2)*voxel_size(3);
    indivi_volume = zeros(1,slice_b-slice_a+1);
    header_row = cell(1,slice_b-slice_a+1);
    
    for slice_i = slice_a:slice_b
        target_slice = (Combo_2D_abnorm(:,:,slice_i));
        indivi_volume(slice_i-slice_a+1) = round(voxel_volume*sum(target_slice(:)),2);       %  transfer to real size later
        header_row{slice_i-slice_a+1} = ['Slice ' num2str(slice_i-1)];
    end
    
    sum_volume = sum(indivi_volume);
    disp('=====================');
    disp('Done');
    
    f = msgbox({'DWMA Quantification Completed';
        ['Volume on slice [' num2str(slice_b-1) ']: ' num2str(round(sum_volume,2))];}); 
end
%
%    end of code
%