
function out_fig = get_smooth(in_fig)

    %  input: 2D DEHSI mask
    %  output: blurred 2D DEHSI mask

    fig = double(in_fig);
    %figure;imagesc(fig);

    figblur = imgaussfilt(fig,2);
    %figure;imagesc(figblur);

    out_fig = figblur>0.3;
    %figure;imagesc(out_fig);