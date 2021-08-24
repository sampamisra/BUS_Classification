clear;

%pre-processed image size
dx = 768;
dy = 495;
%final cropped image size
x = 224; y = 224;
% If you want to get the full image, set x = 350 & y = 495

if x > dx - 10
    x  = dx - 10;
end
if y > dy - 10
    y  = dy - 10;
end

% pixel error compensation
ref = 1; % ref val bigger -> strain image move to right

for idx = 201:201 % min 201 , max 291
    fprintf("\n\n---------------IDX %d Start------------------\n",idx);
    
    input_list = dir(sprintf('cropped/P(%d)/ref*.bmp',idx));
    if length(input_list) <1 % if no ref file in the folder -> skip
        continue
    end
    while(1) 
        close all; 

        C = cast(zeros(x,y,3),'uint8');
        plt_cnt = size(input_list);
        for idx2 = 1:size(input_list)
            filename = sprintf('cropped/P(%d)/%s',idx,input_list(idx2).name);
            A = imread(filename);
            img1 = A(:,18:512,:);
            img2 = A(:,513:1007,:);
            if idx2==1
                C = imfuse(img2,C);
            else
                C = imfuse(img1,C);
            end
        end
        %% Setting ref (pixel compensation)
        while(1)
            %sample images
            figure(); imagesc(img1(200:300,200:300,:)); title(idx);
            figure(); imagesc(img2(200:300,200+ref:300+ref,:)); title(idx);
            in = input("Strain move right? (0-no move, 1-1 pixel move) : ");
            ref = ref + in;
            if in == 0
                break
            end
        end
        %% Setting Starting Point
        while(1)
            fprintf("Select left top start point with Right click!!");
            figure(); imagesc(C); title(idx);
            [iy,ix] = getpts(); % Get the start p
            if ix<1
                ix = 1;
            end
            if iy < 1
                iy = 1;
            end
            ix =  int16(ix);
            iy = int16(iy) ;
            fprintf("\nstart point : ix = %d, iy = %d\n",ix,iy);
            
            tx = ix + x - 1;
            if tx > dx
                tx = dx;
            elseif tx < 1
                tx = 1;
            end
            ty = iy + y - 1;
            if ty > dy
                ty = dy;
            elseif ty < 1
                ty = 1;
            end
            figure; imagesc(C(ix:tx,iy:ty,:)); title('ref');
            savef3 = sprintf('cropped/P(%d)/r_cropped.bmp',idx);
            imwrite(C(ix:tx,iy:ty,:),savef3)
            in = input("Position is right? (0-ok, 1-retry) : ");
            if in == 0
                break
            end
        end
        
        
        for idx2 = 1:size(input_list)
            filename = sprintf('cropped/P(%d)/%s',idx,input_list(idx2).name);
            A = imread(filename);
            img1 = A(:,18:512,:);
            img2 = A(:,513:1007,:);
            % Delete marks to black
            for i = 1:dx
               for j = 1 : dy
                  if img1(i,j,1) == 178 &&img1(i,j,1) == 178 && img1(i,j,3) == 0
                      img1(i,j,:) =cast(zeros(1,1,3),'uint8');
                  end
                  if img1(i,j,1) == 178 &&img1(i,j,1) == 178 && img1(i,j,3) == 178
                      img1(i,j,:) =cast(zeros(1,1,3),'uint8');
                  end
                  if img2(i,j,1) == 178 &&img2(i,j,1) == 178 && img2(i,j,3) == 0
                      img2(i,j,:) = cast(zeros(1,1,3),'uint8');
                  end
                  if img2(i,j,1) == 178 &&img2(i,j,1) == 178 && img2(i,j,3) == 178
                      img2(i,j,:) =cast(zeros(1,1,3),'uint8');
                  end
               end
            end

            img1_r = cast(zeros(x,y,3),'uint8');
            img2_r = cast(zeros(x,y,3),'uint8');
            for i = 1 : dx
                for j = 1 : dy
                   %% Algorithm for Strain Erase Marks
                    if img1(i,j,:) == cast(zeros(1,1,3),'uint8')
                        for rep = 1:4% if there is no color within 3x3, do it again with 5x5
                            cnt = 0;
                            sum=zeros(1,1,3);
                            for k = i - rep : i + rep
                                for r = j - rep : j + rep
                                    if k<=0 || r<=0 || k >dx || r > dy
                                       continue 
                                    end
                                    if img1(k,r,1) ~= 0 || img1(k,r,2) ~= 0 ||img1(k,r,3) ~= 0 
                                        sum = sum + cast(img1(k,r,:),'double');
                                        cnt = cnt + 1;
                                    end
                                end
                            end 
                            if cnt > 0
                                break
                            end
                        end
                        img1_r(i,j,:) = cast( (sum / cnt),'uint8');
                    else
                        img1_r(i,j,:)  = img1(i,j,:) ;
                    end

                   %% Same Algorithm for B-mode
                    if img2(i,j,:) == cast(zeros(1,1,3),'uint8')

                        for rep = 1:4
                            cnt = 0;
                            sum=zeros(1,1,3);

                            for k = i - rep : i + rep
                                for r = j - rep : j + rep
                                    if k<=0 || r<=0  || k >dx || r > dy
                                       continue 
                                    end
                                    if img2(k,r,1) ~= 0 || img2(k,r,2) ~= 0 ||img2(k,r,3) ~= 0 
                                        sum = sum + cast(img2(k,r,:),'double');
                                        cnt = cnt + 1;
                                    end
                                end
                            end
                            if cnt > 0
                                break
                            end
                        end
                        img2_r(i,j,:) = cast( (sum / cnt),'uint8');

                    else
                        img2_r(i,j,:)  = img2(i,j,:) ;
                    end
                end
            end
            
            
            %% Crop Image with ref
            
            tx = ix + x - 1;
            if tx > dx-4
                tx = dx;
            elseif tx < 1
                tx = 1;
            end
            ty = iy + y - 1;
            if ty > dy-4
                ty = dy;
            elseif ty < 1
                ty = 1;
            end
            img1_rr = img1_r(ix:tx,iy:ty,:);
            img2_rr = img2_r(ix:tx,iy+ref:ty +ref,:);

%             subplot(2,plt_cnt,idx2);
%             imshow(img1_rr);
%             title('strain');
%             subplot(2,plt_cnt,idx2+plt_cnt);
%             imshow(img2_rr);
%             title('b-mode');

            savef = sprintf('cropped/P(%d)/d%d_s.bmp',idx,idx2);
            imwrite(img1_rr,savef)
            savef2 = sprintf('cropped/P(%d)/d%d_b.bmp',idx,idx2);
            imwrite(img2_rr,savef2)

        end

        in2 = input("\NextImage(0) or Retry(1)? : ");
        if in2 == 0
            break
        end
        
        
        
    end
 end
