function dwma_display_color(T2w,dwma_bw,head,tail)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 [r,c,n]=size(T2w);
 tmp = zeros(r,c,n);

 
 for i = head:tail
    tmp = dwma_bw(:,:,i);
    roi = imdilate(tmp,strel('disk',1))-tmp;
   %roi = tmp - imerode(tmp,strel('disk',1));
    im = T2w(:,:,i);
    im = im./max(im(:));
    out = imoverlay(im,roi, [1 1 0]);
    %figure,imagesc(flipud(out(:,:,1))),title(int2str(i)); 
    figure,imagesc(rot90(out(:,:,1))),title(int2str(i-1));
%     figure,imshow(rot90(out)),title(int2str(i-1));
    axis image
    axis off
%     ax=gca;
%     ax.PlotBoxAspectRatio = [1 0.5 0.5];
   %figure,imagesc(flipud(out(:,:,1))),title(int2str(i)),title(int2str(i));
 end
 

 

end
