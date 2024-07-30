function bw_out = remove_small_area(bw_in,siz)
L = bwlabel(bw_in,8);
if L==0
    bw_out = bw_in;
else
    stats = regionprops(L,'Area');
    % 
    area = cell2mat(struct2cell(stats));
    ind = find(area>=siz);
    [r,c] = size(bw_in);

    bw_out =zeros(r,c);
    for i = 1:length(ind)
        bw_out(L==ind(i))=1;
    end
end
