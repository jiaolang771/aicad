
%
%   DEHSI segmentation and detection
%
function DEHSI_both

%%    segmentation
%-------------------------%
% Check for SPM12 %
%-------------------------%



[spmpath,~,~] = fileparts(which('spm'));
if ~strcmp('spm12',spmpath(end-4:end))
    warndlg('SPM12 must be set in MATLAB path.','Preprocessing Aborted')
    return
end

%addpath(fullfile(curr_path,'\func\'));
%-----------------------------------%
% Select subjects/compile file list %
%-----------------------------------%
% Get directories of subjects to preprocess
subdirs = uigetdir(pwd,'Select brain MRI image folder');
if subdirs==0
    return
end

disp('=====================');
disp('Running...');

% Get unique string for each scan type
% dtype = inputdlg([{'Structural Image'}],...
%     'Enter a unique string for the file',repmat([1 70],1,1),{'T2'});
%     dext= inputdlg([{'Structural Image'},],...
%         'Enter the data format: .nii or .img',repmat([1 100],1,1));
dtype = {'T2'};
dext = {'.nii','.img'};
flist = struct;
nsubs = 1;
sub = 0;
% for n=1:nsubs
n=1;
s = strsplit(subdirs,filesep);
files = cellstr(strtrim(ls(subdirs))); %strsplit(cellstr(strtrim(ls(subdirs{n}))));
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

%end
if isempty(fieldnames(flist))
    return
end


%-----------------------------------------%
% Write the SPM preprocessing batch files %
%-----------------------------------------%
filelist = cell(nsubs,1);
% [tmpltn,tmpltp]=uigetfile('*.nii','Select template for normalization',fullfile(spmpath,'TPM'));

curr_path = pwd;
%-----------------------------------------%
% Write the T2w batch files %
%-----------------------------------------%


for x=1:length(flist)
    if isempty(flist(x).(dtype{1}))
        break %continue    % if no structural file exists for subject x, skip subject
    end
    [apath, aname, ext] = fileparts(fullfile(char(flist(x).dir),char(flist(x).(dtype{1}))));
    
    %%   automatic AC-PC alignment
    %p=fullfile(char(flist(x).dir),char(flist(x).(dtype{1})));
    %auto_reorient(p);
    
    filelist{x} = fullfile(apath,['SPMpreproc_neonatal_' char(flist(x).subid) '.m']);
    fid = fopen(filelist{x},'w');
    j = 0;
    
    
    %% Script batch job for Unified Segmentation  job 1  old segmentation
    j = j + 1;
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.data = {''' apath filesep aname ext '''};']);
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.GM = [0 0 1];'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.WM = [0 0 1];'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.CSF = [0 0 1];'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.biascor = 1;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.cleanup = 0;']);
    
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.tpm  = {'],...
        [''''  fullfile(curr_path,'\func\neo_atlas\infant-neo-seg-gm.nii,1') ''''],...
        [''''  fullfile(curr_path,'\func\neo_atlas\infant-neo-seg-wm.nii,1') ''''],...
        [''''  fullfile(curr_path,'\func\neo_atlas\infant-neo-seg-csf.nii,1') ''''],...
        '};');
    
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.ngaus=[2'],...
        '2',...
        '2',...
        '4];');
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.regtype = '''';'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.warpreg = 1;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.warpco = 25;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.biasreg = 0.0001;']);
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.biasfwhm = 60;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.samp = 3;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.msk = {''''};']);
    
    
    %% Script batch job to create brain extracted image in subject-space    job2   image calculation
    j = j + 1;
    fprintf(fid,'%s\n',['matlabbatch{' num2str(j) '}.spm.util.imcalc.input = {'],...
        ['''' fullfile(apath,['c1' aname,ext]) ',1'''],...
        ['''' fullfile(apath,['c2' aname,ext]) ',1'''],...
        ['''' fullfile(apath,['c3' aname,ext]) ',1'''],...
        ['''' fullfile(apath,['m' aname,ext]) ',1'''],...
        '};');
    fprintf(fid,'%s\n',['matlabbatch{' num2str(j) '}.spm.util.imcalc.output = ''' fullfile(apath,['bs_' aname,ext]) ''';']);
    fprintf(fid,'%s\n',['matlabbatch{' num2str(j) '}.spm.util.imcalc.outdir = {''''};']);
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.util.imcalc.expression = ''(i4.*((i1+i2+i3)>0.7))'';'],...
        ['matlabbatch{' num2str(j) '}.spm.util.imcalc.var = struct(''name'', {}, ''value'', {});'],...
        ['matlabbatch{' num2str(j) '}.spm.util.imcalc.options.dmtx = 0;'],...
        ['matlabbatch{' num2str(j) '}.spm.util.imcalc.options.mask = 0;'],...
        ['matlabbatch{' num2str(j) '}.spm.util.imcalc.options.interp = 1;'],...
        ['matlabbatch{' num2str(j) '}.spm.util.imcalc.options.dtype = 4;']);
    
    %% Script batch job for second segmentation after skull striping    job3  old segmentation
    j = j + 1;
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.data = {''' apath filesep 'bs_' aname ext '''};']);
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.GM = [0 0 1];'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.WM = [0 0 1];'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.CSF = [0 0 1];'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.biascor = 1;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.output.cleanup = 0;']);
    
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.tpm  = {'],...
        [''''  fullfile(curr_path,'\func\neo_atlas\infant-neo-seg-gm.nii,1') ''''],...
        [''''  fullfile(curr_path,'\func\neo_atlas\infant-neo-seg-wm.nii,1') ''''],...
        [''''  fullfile(curr_path,'\func\neo_atlas\infant-neo-seg-csf.nii,1') ''''],...
        '};');
    
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.ngaus=[2'],...
        '2',...
        '2',...
        '4];');
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.regtype = '''';'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.warpreg = 1;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.warpco = 25;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.biasreg = 0.0001;']);
    fprintf(fid,'%s\n',...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.biasfwhm = 60;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.samp = 3;'],...
        ['matlabbatch{' num2str(j) '}.spm.tools.oldseg.opts.msk = {''''};']);
    
    
    %%    close the file
    
    
    fclose(fid);
end


%---------------------------%
% Execute SPM preprocessing %
%---------------------------%

nsubs = length(filelist);
if nsubs==0
    warndlg('SPM12 must be set in MATLAB path.','Preprocessing Aborted')
    return
elseif nsubs==1
    ispar = 0;
end
%% Execute batch files
for x=1:nsubs
    if ~isempty(filelist{x})
        spm_jobman('initcfg');
        spm('defaults', 'FMRI');
        spm_jobman('run', filelist{x}, cell(0, 1));
    end
end

%%
%-----------------------------------------%
% Read processed files and detect abnormality %
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
    Brain = Brain_struct.img;                                   %  brain image
    
    GM_struct = load_untouch_nii(fullfile(apath,['c1bs_' aname ext]));
    GM = GM_struct.img;                                   %  GM probability map
    
    WM_struct = load_untouch_nii(fullfile(apath,['c2bs_' aname ext]));
    WM = WM_struct.img;                                   %  WM probability map
    
    CSF_struct = load_untouch_nii(fullfile(apath,['c3bs_' aname ext]));
    CSF = CSF_struct.img;
    
      
    %%   set up parameters
    [~, ~, brain_Z] = size(Brain);
    
    dpara = inputdlg([{'DWMA intensity cutoff'},{'White matter probability map threshold'},{'Erosion voxel'},{'Small region removal'},...
        {'Interested slice start from'},{['Interested slice to (no more than ' num2str(brain_Z-1) ')']}],...
        'Enter parameter for DWMA detection',repmat([1 100],6,1),...
        [{'1.8'},{'0.95'},{'10'},{'10'},{'20'},{'80'}]);
    
    if isempty(dpara)
        return
    end
    
    std_coff = str2double(dpara{1});
    pm_cutoff = str2double(dpara{2});
    erosion_cutoff= str2double(dpara{3});
    small_cutoff= str2double(dpara{4}); 
    slice_a= str2double(dpara{5})+1;
    slice_b= str2double(dpara{6})+1;
      
    
    %%   apply probability map
    
    seg_cutoff = pm_cutoff;
    WM_pm_bin = ((double(WM)/255)>seg_cutoff)&(WM>GM)&(WM>CSF);  %   WM_pm_bin:  high confidence pm
    GM_pm_bin = ((double(GM)/255)>seg_cutoff)&(GM>WM)&(GM>CSF);
    CSF_pm_bin = ((double(CSF)/255)>seg_cutoff)&(CSF>WM)&(CSF>GM);
    Combo_pm_bin = GM_pm_bin|WM_pm_bin;   %  GM+WM  or WM probability map
   
    WM_pm_all = (WM>GM)&(WM>CSF)&(WM>0);    %  all pm w/o pm cutoff
    GM_pm_all = (GM>WM)&(GM>CSF)&(GM>0);
    CSF_pm_all = (CSF>WM)&(CSF>GM)&(CSF>0);
    Brain_pm_all = WM_pm_all | GM_pm_all | CSF_pm_all;
%     disp(sum(CSF_pm_bin(:)));
%     disp(sum(CSF_pm_all(:)));
    
    
    %%   transfer WM+GM to Z-score normalization
    
    [x_ind,y_ind,z_ind]=size(Brain);
    nBrain = zeros(size(Brain));
    nCombo = zeros(size(Brain));
    for i=1:z_ind
        
        Combo_pm_2D = Combo_pm_bin(:,:,i);
        Brain_pm_2D = Brain_pm_all(:,:,i);
        
        Brain_single = Brain(:,:,i);
        
        Combo_2D_data=Brain_single(Combo_pm_2D);
        Combo_2D_mean = mean(Combo_2D_data);
        Combo_2D_std = std(Combo_2D_data);
        
        Brain_2D_data=Brain_single(Brain_pm_2D);
        Brain_2D_mean = mean(Brain_2D_data);
        Brain_2D_std = std(Brain_2D_data);
        for j=1:x_ind
            for k=1:y_ind
                if Combo_pm_2D(j,k)
                    nCombo(j,k,i)=(Brain(j,k,i)-Combo_2D_mean)/Combo_2D_std;
                end
                if Brain_pm_2D(j,k)
                    nBrain(j,k,i)=(Brain(j,k,i)-Brain_2D_mean)/Brain_2D_std;
                end
            end
        end
    end
    
    %% WM+GM Combo abnormality detection
    
    [Combo_X, Combo_Y, Combo_Z] = size(Combo_pm_bin);
    Combo_2D_abnorm = false([Combo_X, Combo_Y, Combo_Z]);
    
    
    for combo_i=1:Combo_Z
        WM_pm_2D = WM_pm_bin(:,:,combo_i);
        Combo_pm_2D = Combo_pm_bin(:,:,combo_i);
        
        Combo_2D_single = nCombo(:,:,combo_i);
        Combo_2D_data = Combo_2D_single(Combo_pm_2D);
        Combo_2D_mean = mean(Combo_2D_data);
        Combo_2D_std = std(Combo_2D_data);
        
        Combo_2D_abnorm(:,:,combo_i) = (Combo_2D_single.*WM_pm_2D)>=(Combo_2D_mean+Combo_2D_std*std_coff);
        %figure;imagesc(WM_2D_abnorm(:,:,wm_i))
    end
 
    %%  image erode

    se = strel('disk',erosion_cutoff);
    Combo_mask = WM+GM;
    for iii=1:Combo_Z
        origI = Combo_mask(:,:,iii);
       % origI =imfill(origI,'holes');
        erodedI = imerode(origI,se);       
        Mask = erodedI>0;
        %figure; imagesc(Mask);
        Combo_2D_abnorm(:,:,iii)=Combo_2D_abnorm(:,:,iii).*Mask;
        nCombo(:,:,iii) = nCombo(:,:,iii).*Mask;
    end
    
    
    %%   remove small regions and fill holes
    
    for iii=1:Combo_Z
        Combo_2D_abnorm(:,:,iii)=remove_small_area(Combo_2D_abnorm(:,:,iii),small_cutoff);
        Combo_2D_abnorm(:,:,iii)=imfill(Combo_2D_abnorm(:,:,iii),'holes');
    end
    

    
end

%%   calculate DEHSI volume

voxel_size = Brain_struct.hdr.dime.pixdim(2:4);
voxel_volume = voxel_size(1)*voxel_size(2)*voxel_size(3);

indivi_dwma_volume = zeros(1,slice_b-slice_a+1);
indivi_WM_volume = zeros(1,slice_b-slice_a+1);
indivi_GM_volume = zeros(1,slice_b-slice_a+1);
indivi_CSF_volume = zeros(1,slice_b-slice_a+1);
header_row = cell(1,slice_b-slice_a+1);

for slice_i = slice_a:slice_b
    
    target_slice = (Combo_2D_abnorm(:,:,slice_i));
    target_slice_WM = (WM_pm_all(:,:,slice_i));
    target_slice_GM = (GM_pm_all(:,:,slice_i));
    target_slice_CSF = (CSF_pm_all(:,:,slice_i));
    
    indivi_dwma_volume(slice_i-slice_a+1) = round(voxel_volume*sum(target_slice(:)),2);       %  transfer to real size later
    indivi_WM_volume(slice_i-slice_a+1) = round(voxel_volume*sum(target_slice_WM(:)),2); 
    indivi_GM_volume(slice_i-slice_a+1) = round(voxel_volume*sum(target_slice_GM(:)),2); 
    indivi_CSF_volume(slice_i-slice_a+1) = round(voxel_volume*sum(target_slice_CSF(:)),2); 
    
    header_row{slice_i-slice_a+1} = ['Slice ' num2str(slice_i-1)];
end

sum_volume_dwma = sum(indivi_dwma_volume);
sum_volume_WM = sum(indivi_WM_volume);
sum_volume_GM = sum(indivi_GM_volume);
sum_volume_CSF = sum(indivi_CSF_volume);

%output_vol = [indivi_volume,sum_volume];

%%  save to file

% header_row = [header_row,'Sum'];
% output_file = {' ';'Volume'};
% output_file = [output_file,[header_row;num2cell(output_vol)]];
% xlswrite(fullfile(apath,[aname,'_volume.xlsx']),output_file);


save(fullfile(apath,[aname '_' num2str(std_coff) '_DWMA_mask.mat']),'Combo_2D_abnorm', 'nBrain', 'nCombo');

save(fullfile(apath,[aname '_BRAIN.mat']),...
    'Brain','WM_pm_bin','GM_pm_bin','CSF_pm_bin','voxel_size', 'WM_pm_all','GM_pm_all','CSF_pm_all');


%%    display
disp('=====================');
disp('Done');


% display a contour for a subject
dwma_display(Brain,Combo_2D_abnorm,slice_a,slice_b);

f = msgbox({'DWMA Quantification Completed';
    ['DWMA volume on slice [' num2str(slice_a-1) '~' num2str(slice_b-1) ']: ' num2str(round(sum_volume_dwma,2))];
    ['White matter volume on slice [' num2str(slice_a-1) '~' num2str(slice_b-1) ']: ' num2str(round(sum_volume_WM,2))];
    ['Grey matter volume on slice [' num2str(slice_a-1) '~' num2str(slice_b-1) ']: ' num2str(round(sum_volume_GM,2))];
    ['CSF volume on slice [' num2str(slice_a-1) '~' num2str(slice_b-1) ']: ' num2str(round(sum_volume_CSF,2))];
    });

%
%    end of code
%
