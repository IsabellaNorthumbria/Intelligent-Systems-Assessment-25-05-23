%clear workspace
clear; 

%ask user what video they want to choose
[filename, pathname] = uigetfile({'*.mpg;*.mp4;*.avi;*.wmv', 'Video files (*.mpg,*.mp4,*.avi,*.wmv)'}, 'Pick a video file');
file=VideoReader([pathname,filename]); 
fileFrame= readFrame(file);


%DETECTING HAND 

%image segmentation to detect areas that have skin color
% hand=imread(fileFrame);
    hand=fileFrame;
    hand=rgb2ycbcr(hand);
    for i=1:size(hand,1)
        for j= 1:size(hand,2)
            cb = hand(i,j,2);
            cr = hand(i,j,3);
            if(~(cr > 132 && cr < 173 && cb > 76 && cb < 126))
                hand(i,j,1)=235;
                hand(i,j,2)=128;
                hand(i,j,3)=128;
            end
        end
    
    end
 hand=ycbcr2rgb(hand);   
subplot(2,2,1);
    image1=imshow(hand);
    axis on;
    title('Image segmentation');
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
%convert segmented image to grayscale by using the grayscale function
    grayScale=rgb2gray(hand);
    subplot(2,2,2);
    image2=imshow(grayScale);
    axis on;
    title('Grayscale');
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
%convert grayscale to binary image by using the binary function
    binary = imbinarize(grayScale);
    subplot(2, 2, 3);
    axis on;
    image3=imshow(binary);
    title('Binary');
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
%Experiment 3 code (line 49): make the boundary box to box any black figures 
    binary = binary == 0;

%lable each connected component a unique label
labeledImage = bwlabel(binary);  
measurements = regionprops(labeledImage, 'BoundingBox', 'Area');
for k = 1 : length(measurements)
      thisBB = measurements(k).BoundingBox;
      rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
      'EdgeColor','r','LineWidth',2 )
end

%extract hand
allAreas = [measurements.Area];
[sortedAreas, sortingIndexes] = sort(allAreas, 'descend');
%hand is the second biggest hence why you put '2' 
handIndex = sortingIndexes(2); 
%get hand from labeled image 
handImage = ismember(labeledImage, handIndex);


handImage = handImage > 0;
subplot(2, 2, 4);
image4=imshow(handImage, []);
title('Hand detected');
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
%End of detecting hand


%detect hand in frames and return a bounding box on where the hand is
bboxes=step(handImage ,fileFrame);
fileFrame= insertShape(fileFrame,"Rectangle",bboxes,'Color','yellow');
figure; imshow(fileFrame); title("Detected hand with KLT");


%KLT tracking

%coverts first box into a listing of 4 points
pointsbboxes= bboxes2points(bboxes(1, :));

%detect and display features of hand
points = detectMinEigenFeatures(handImage(fileFrame), "ROI", bboxes);
figure, imshow(fileFrame), hold on, title ("Detected feature points");
plot(points);

%start tracker to track points
tracker = vision.PointTracker("MaxBidirectionalError", 2);
points = points.Location;
initialize(pointTracker,points,fileFrame);

%show results to video player
videoPlay = vision.video("Position",...
    [100 100 [size(fileFrame, 2), size(fileFrame, 1)]+30]);

%tracking the hand
trackPoints = points;
%loop to detect and track hand to each frame
while hasFrame(file)
    %get next frame
    fileFrame = readFrame(file);

    %track points
    [points, isFound] = step(tracker, fileFrame);
    pointsVisible = points(isFound, :);
    inliers = trackPoints(isFound, :);

    %KLT tracking needs at least 2 points
    if size(pointsVisible, 1) >=2 
        %find the geometric transformation by comparing points from
        %previous frame
        [xform, inlierIdx] = estimateGeometricTransform2D(...
            inliers, pointsVisible, "similarity", "MaxDistance",4);
        inliers = inliers(inlierIdx, :);
        pointsVisible = pointsVisible(inlierIdx, :);
        
        % move box points to where the hand is 
        pointsbboxes = transformPointsForward(xform,pointsbboxes);

        %show box around tracked hand
        bboxPolygon = reshape(pointsbboxes',1,[]);
        fileFrame = insertShape(fileFrame,"Polygon", bboxPolygon, ... 
            "LineWidth",2);

        %show tracked points
        fileFrame = insertMarker(fileFrame, pointsVisible, "+", ...
            "Color", "white"); 
        %reset points
        trackPoints = pointsVisible;
        setPoints(tracker,trackPoints);
    end
    %show video with tracked hand
    implay(videoPlay,fileFrame);
    
end

%clean up!
release(videoPlay);
release(tracker);



