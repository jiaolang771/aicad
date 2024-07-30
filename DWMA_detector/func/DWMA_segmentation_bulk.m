%
function DWMA_segmentation_bulk
% Neonatal Brain White Matter Abnormality Detection
%
% This function is designed for the following data organization structure:
%   -StudyID (e.g. IRC001H)
%       -Data
%           -SubjectID 1
%           -...
%           -SubjectID N
%
% SPMpreproc is currently compatible with SPM12 only.
%
% Developed by: Hailong Li
% Last updated: Dec-10-2019
%-------------------------%
% Check for SPM12 %
%-------------------------%
[spmpath,~,~] = fileparts(which('spm'));
if ~strcmp('spm12',spmpath(end-4:end))
    warndlg('SPM12 must be set in MATLAB path.','Preprocessing Aborted')
    return
end
addpath(genpath([pwd '\func\']));

%-----------------------------------%
% Select subjects/compile file list %
%-----------------------------------%
% Get directories of subjects to preprocess
subdirs = uigetdirs(pwd);
if isempty(subdirs)
    return
end

% Get unique string for each scan type
% dtype = inputdlg([{'Structural Image'},],...
%     'Enter a unique string for the file',repmat([1 70],1,1));
%     dext= inputdlg([{'Structural Image'},],...
%         'Enter the data format: .nii or .img',repmat([1 100],1,1));
dtype = {'T2'};
dext = {'.nii','.img'};
flist = struct;
nsubs = length(subdirs);
sub = 0;
for n=1:nsubs
    s = strsplit(subdirs{n},filesep);
    files = cellstr(strtrim(ls(subdirs{n}))); %strsplit(cellstr(strtrim(ls(subdirs{n}))));
   
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
    
    %% Check for structural file
    check = ~cellfun(@isempty,regexp(files,dtype{1}));
    if sum(check)==0
        h = warndlg({'Error: no structural file found';'';...
            ['Please check ' subdirs{n} ' for complete data.'];...
            '';'Subject will be skipped.'});
        uiwait(h);
        continue
    elseif sum(check)==1
        sub = sub + 1;
        flist(n).subid = s{end};
        flist(n).dir = subdirs{n};
        flist(sub).(dtype{1}) = files(check);
    else
        temp = files(check);
        [sel,~]=listdlg('PromptString',['Multiple ' dtype{1} ' files found.  ' ...
            'Select file for preprocessing'],...
            'SelectionMode','single','ListSize',[478 200],'ListString',temp);
        if isempty(sel)
            h = warndlg({'No structural file selected.';'';...
                ['Subject ' s{end} ' will be skipped']});
            uiwait(h);
            continue
        else
            sub = sub + 1;
            flist(n).subid = s{end};
            flist(n).dir = subdirs{n};
            flist(sub).(dtype{1}) = temp(sel);
        end
    end
    
end
if isempty(fieldnames(flist))
    return
end
%%  save file list
% save('subject_file_list.mat','flist','dtype');


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



