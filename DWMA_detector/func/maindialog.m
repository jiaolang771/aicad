function  maindialog

    d = dialog('Position',[500 350 600 300],'Name','DWMA detector (version beta)');

    txt1 = uicontrol('Parent',d,...
               'Style','text',...
               'Position',[60 60 500 200],...
               'String','Welcome to DWMA Detector',...
               'ForegroundColor','b',...
               'FontSize',25);
           
    txt2 = uicontrol('Parent',d,...
               'Style','text',...
               'Position',[50 110 500 100],...
               'String',{'Neurodevelopmental Disorders Prevention Center';...
               'Cincinnati Children''s Hospital Medical Center';...
               'www.cincinnatichildrens.org'},...
               'FontSize',10);
           
    txt3 = uicontrol('Parent',d,...
               'Style','text',...
               'Position',[50 30 500 100],...
               'String',{'Choose tasks:'},...
               'FontSize',15);
     
    persistent choice;
    if isempty(choice)
        choice = 'Segmentation Only';
    end
    %press = 0;       
   
    
    popup = uicontrol('Parent',d,...
           'Style','popup',...
           'Position',[200 70 200 25],...
           'String',...
           {'Segmentation Only (Single)';...
           'Segmentation Only (Bulk)';...
           'Detection Only';...
           'Segmentation&Detection (Single)';...
           'Manual Correction';...
           'DWMA Quantification';...
           'Whole Brain Tissue Volume'},...
           'Callback',@popup_callback);
       
    
    btn = uicontrol('Parent',d,...
               'Position',[260 20 70 25],...
               'String','Start',...
               'Callback',@btn_callback);
           
        
    uiwait(d);
    
    function popup_callback(popup,event)
          idx = popup.Value;
          popup_items = popup.String;
          choice = char(popup_items(idx,:));
    end
 
    function btn_callback(btn,event)
           %delete(gcf);  
           num = strlength(choice);
           %press=press+1;
           %if press>0
           if num== 26
               disp('Start Segmentation (Single)');
               DEHSI_seg;
           elseif num==24
               disp('Start Segmentation (Bulk)');
               DWMA_segmentation_bulk;
           elseif num==14
               disp('Start Detection Only');
               DEHSI_dec;
           elseif num==31
               disp('Segmentation & Detection');
               DEHSI_both;
           elseif num==17   
               disp('Manual Correction');
               DEHSI_draw;
           elseif num==19
               disp('DWMA Quantification');
               DEHSI_show;
           elseif num==25
               disp('Whole Brain Tissue Volume');
               DEHSI_tissue;
               
           end
           %end
           
           %display(press);
           
           %uiresume(gcbf);
     end
    
end