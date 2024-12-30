%% Octava function implementation
clc; close all; clear;
function_path_1 = "D:\DTU\Bachelor_data\Matlab\MatLab\Functions";
% File paths
OCTA_folder_path = "D:\DTU\Bachelor_data\Stimulation\Dynamics\Baseline\OCTA\Dynamic_Baseline_Part_07_D.dcm";
OCT_folder_path = "D:\DTU\Bachelor_data\Stimulation\Dynamics\Baseline\OCT\Dynamic_Baseline_Part_07_S.dcm";

% Read data and metadata
OCTA_data = dicomread(OCTA_folder_path);
meta_info = dicominfo(OCT_folder_path);

% Extract pixel spacings
axial_spacing = meta_info.PixelSpacing(1); % Spacing in the X direction (mm)
lateral_spacing = meta_info.SpacingBetweenSlices;   % Spacing in the Y direction (mm)

% Create Maximum Intensity Projection (MIP)
mip = max(OCTA_data, [], 1);  % Max projection along the first dimension
mip = squeeze(mip);           % Remove singleton dimension

MIP_OCTA = imresize(mip, [1366,1366], 'bilinear');

% Define physical dimensions for axes
x = (0:120 - 1) * lateral_spacing; % Lateral (X) dimension
y = (0:1366 - 1) * axial_spacing;   % Axial (Z) dimension

% Display the resized MIP
figure;
imagesc(x,y,MIP_OCTA);
axis equal;
xlabel('Lateral Distance (mm)');
ylabel('Axial Distance (mm)');
colormap('gray'); % Grayscale colormap
ylim([0,6]);
xlim([0,6]);

% Step 1: Threshold the image to identify flow regions
% Use Otsu's method or define a threshold empirically
threshold = graythresh(MIP_OCTA) * max(MIP_OCTA(:));  % Otsu's method
flowRegion = MIP_OCTA > threshold;  % Binary mask for flow regions

% Step 2: Calculate Flow Index (mean intensity in flow regions)
flowIntensityValues = MIP_OCTA(flowRegion);  % Intensities in flow regions
flowIndex = mean(flowIntensityValues);   
max_val = 65535;
flowScore = flowIndex/max_val;
disp(['Flow Index: ', num2str(flowIndex)]);
disp(['Flow Score: ', num2str(flowScore)]);

%% From octava 
cd(function_path_1)
% clc; close all;
% fol = function_path_2;
% Set up segmentation parameters (adjust these to your needs)
SegmentationMethod = 'Fuzzy means';  % Options: 'Fuzzy means', 'Adaptive thresholding'
MedfiltFlag = false;                  % Apply median filter if true
FrangiFlag = true;                   % Use Frangi filter if true
kernelSize = 3;                      % Kernel size for median filtering
sigmaMin = 0.5;                      % Frangi filter minimum sigma
sigmaMax = 10;                        % Frangi filter maximum sigma
ATkernelSize = 15;                   % Kernel size for adaptive thresholding
TwigSize = 20;                       % Minimum branch length for skeletonization
bin_overlay = 'jet';                 % Colormap for visualization
dispFlag = true;                     % Display and save images if true
FileLabelName = 'ImageOutput';       % Base name for saved images

inputImage = MIP_OCTA;  

Im = otsu_and_binarize_image(MIP_OCTA);
Im = double(Im);

[y] = max(inputImage(2:end,2:end));
[~,cc] = max(y);
[~,rr] = max(inputImage(2:end,cc));

cc=cc+1;
rr=rr+1;


% Step 1: Convert to double and normalize to [0, 1] range

if max(Im(:)) > 1  % Normalize only if not already in [0,1]
    Im = Im / max(Im(:));
end
% Display the result of preprocessing
figure;
subplot(1,2,1);
imshow(Im);
title('Preprocessed Image for Frangi Filter');

% Step 3: Apply the Frangi filter with modified parameters
if FrangiFlag
    opts.sigmarange = [15, 80];       % Equivalent to np.linspace(1, 80, 6) in Python
    opts.sigmastepsize = 4;         % Number of steps (you can adjust this for finer scales)
    opts.correctionconst1 = 0.8;     % Equivalent to alpha in Python
    opts.correctionconst2 = 15;     % Equivalent to beta in Python
    
    vesselEnhancedImage = frangi_2Dfilter(Im, opts); % Applying Frangi filter
    % Normalize Frangi output to [0, 1]
    vesselEnhancedImage = vesselEnhancedImage / max(vesselEnhancedImage(:));

    % Verify dimensions
    if ~isequal(size(vesselEnhancedImage), size(Im))
        error('Frangi output and original image dimensions do not match.');
    end
end



% Display the result of Frangi filtering
subplot(1,2,2);
imshow(vesselEnhancedImage, []);
title('After Frangi Filter');

%% Skeletonize

% Binarize based on selected segmentation method
if strcmp(SegmentationMethod, 'Fuzzy means')
    nth = 2; % Number of sets for fuzzy thresholding
    BinaryImage = fuzzy_thresholding(vesselEnhancedImage, nth, 3) - 1;

elseif strcmp(SegmentationMethod, 'Adaptive thresholding')
    BinaryImage = double(adaptivethresholding(Im, ATkernelSize));
else
    error('Incorrect segmentation method selected');
end

% Generate skeletonized image
BinaryImage = imclose(BinaryImage, strel('disk', 1));

[Nx, Ny] = size(BinaryImage);
L = logical(BinaryImage);


sk2 = bwmorph(L, 'skel', Inf);

% Remove branches smaller than TwigSize
sk2 = bwskel(sk2, 'MinBranchLength', TwigSize);
% sk2 = imdilate(sk2, strel('disk', 1));

% figure;
dilated_skel = imdilate(sk2, strel('disk', 2));
imshow(dilated_skel, []); % Display the image, with intensity scaling for better visualization
colormap(gray); % Set the colormap to grayscale
title('Skeletonized Image');


%% plot skeleton overlay

% Overlay skeleton on Frangi filter result
figure;
imshow(MIP_OCTA, []);
hold on;

% Use a colored overlay for the skeleton
skeletonOverlay = cat(3, dilated_skel, zeros(size(dilated_skel)), zeros(size(dilated_skel))); % Red channel only
h = imshow(skeletonOverlay);

% Adjust transparency
set(h, 'AlphaData', dilated_skel * 0.8); % Adjust transparency factor
title('Skeletonized Image Overlayed on Frangi Filter');

%% Network Analysis (Skeleton to Graph Conversion)

if sum(sk2(:)) == 0
    nodes = [];
    links = [];
else
    [~, nodes, links] = Skel2Graph3D(sk2, TwigSize);
    wl = sum(cellfun('length', {nodes.links}));

    % Network refinement using iterations
    maxIter = 5;
    for iter = 1:maxIter
        skel2 = Graph2Skel3D(nodes, links, Nx, Ny, 1);
        [~, nodes, links] = Skel2Graph3D(skel2, 0);
        wl_new = sum(cellfun('length', {nodes.links}));
        if wl_new == wl
            break;
        end
        wl = wl_new;
    end
end


% Add parameters to links structure that are required for calculating metrics
for i = 1:length(links)
    % Add x and y coordinates for each point on the link
    [links(i).x, links(i).y, ~] = ind2sub([Nx, Ny, 1], links(i).point); 
    
    % Calculate the pixel length of the branch
    d = diff([links(i).y', links(i).x']);
    links(i).pixLength = sum(sqrt(sum(d .* d, 2))); % Length of branch in pixels
    
    % Calculate the end-to-end (arc) length of the branch
    links(i).end2endLength = sqrt((links(i).x(end) - links(i).x(1))^2 + ...
                                  (links(i).y(end) - links(i).y(1))^2);
                              
    % Calculate tortuosity as the ratio of pixLength to end-to-end length
    links(i).tortuosity = links(i).pixLength / links(i).end2endLength;
    
    % Measure diameter at each point within the link and calculate mean diameter
%     links(i).diameterMeasurements = D(links(i).point); % Diameter measurements
%     links(i).diameter = mean(links(i).diameterMeasurements); % Mean diameter
end

% Display the network overlay
disp('Starting network overlay plotting');

% Display the original image in grayscale as the background
fig = figure;
fig.Visible = 'on';
figAxes = axes(fig);
imshow(inputImage, [], 'Parent', figAxes); 
hold(figAxes, 'on');
axis(figAxes, [0, size(inputImage, 2), 0, size(inputImage, 1)]);
axis(figAxes, 'equal');

% Plot nodes and links with different colors based on type
for i = 1:length(nodes)
    x1 = nodes(i).comx;
    y1 = nodes(i).comy;

    for j = 1:length(nodes(i).links) % Draw all connections of each node
        if nodes(nodes(i).conn(j)).ep == 1
            col = 'g'; % branches are green
        else
            col = 'y'; % links are yellow
        end
        if nodes(i).ep == 1
            col = 'g'; % override to green if current node is endpoint
        end
        if nodes(nodes(i).conn(j)).ep == 1 && nodes(i).ep == 1
            col = 'b'; % both nodes are endpoints, color blue
        end

        % Draw edges as lines using voxel positions
        plot(figAxes, links(nodes(i).links(j)).y, links(nodes(i).links(j)).x, 'Color', col, 'LineWidth', 2);
    end

    % Draw nodes as circles
    if nodes(i).ep == 1
        ncol = 'c'; % cyan for endpoints
    else
        ncol = 'm'; % magenta for regular nodes
    end
    plot(figAxes, y1, x1, 'o', 'Markersize', 4, 'MarkerFaceColor', ncol, 'Color', 'k');
end

% Finalize display properties
axis image; axis off;
drawnow;
fig.Position = fig.Position .* [1 1 5 5];
disp('Network overlay plotted.');

%% Branching angle calculation - linear regression
% clc;close all;
% Initialize an array to store angles at junction points
junctionAngles = [];
N = 4; % N >= 3 
figure;
% Loop through all nodes to identify regular (non-endpoint) nodes
for i = 1:length(nodes)
    % Check if the current node is a regular junction (not an endpoint)
    if nodes(i).ep == 0  % ep == 0 means it's a regular junction node
        % Get the coordinates of the current junction node
        x1 = nodes(i).comx;
        y1 = nodes(i).comy;

        % Get the indices of links connected to this node
        connectedLinks = nodes(i).links;
        numLinks = length(connectedLinks);

        % Only process if the junction has at least two connected links
        if numLinks >= 2
            % Store direction vectors for all branches at this junction
            branchVectors = zeros(numLinks, 2);

            % Get direction vectors for all branches
            for j = 1:numLinks
                link = links(connectedLinks(j));
                [x_points, y_points] = getFirstNPoints(link, x1, y1, N);
                branchVectors(j, :) = getDirectionVector(x_points, y_points, x1, y1);
            end

            % Calculate angles of the vectors with respect to the x-axis
            vectorAngles = atan2d(branchVectors(:,2), branchVectors(:,1));
            % Ensure angles are in the range [0, 360)
            vectorAngles = mod(vectorAngles, 360);

            % Sort the angles and corresponding vectors
            [sortedAngles, sortIdx] = sort(vectorAngles);
            branchVectors = branchVectors(sortIdx, :);

            % Compute internal angles between adjacent vectors
            angles = zeros(numLinks, 1);
            for j = 1:numLinks
                % Use circular indexing to get the next branch
                idx1 = j;
                idx2 = mod(j, numLinks) + 1;
                angle1 = sortedAngles(idx1);
                angle2 = sortedAngles(idx2);

                % Compute the internal angle between adjacent vectors
                internalAngle = angle2 - angle1;
                if internalAngle < 0
                    internalAngle = internalAngle + 360;
                end

                angles(j) = internalAngle;
            end

            % Store the results
            for j = 1:numLinks
                junctionAngles = [junctionAngles; i, angles(j)];
            end

%             % Optional: Visualize the junction and angles
            
            if i < 3
                subplot(2,1,i)
                hold on;
                for j = 1:numLinks
                    % Plot each branch and direction vector
                    link = links(connectedLinks(sortIdx(j)));
                    [x_points, y_points] = getFirstNPoints(link, x1, y1, N);
                    plot(y_points, x_points, '.', 'MarkerSize', 30);
                    quiver(y1, x1, branchVectors(j, 2)*10, branchVectors(j, 1)*10, 0, 'LineWidth', 2);
                end
                title(['Junction ' num2str(i) ' - Branch Angles']);
                grid on;
                axis equal;
            end
        end
    end
end

% % Display results
% disp('Junction Node Index and Calculated Angles (in degrees):');
% disp(junctionAngles);

%% Find minimum angle for every node

% Initialize an array to store the smallest angle for each node
uniqueNodes = unique(junctionAngles(:,1));  % Get unique node indices
minAngles = [];  % Array to store minimum angle for each node

% Loop through each unique node to find the smallest angle
for i = 1:length(uniqueNodes)
    nodeIndex = uniqueNodes(i);
    
    % Extract all angles for this node
    nodeAngles = junctionAngles(junctionAngles(:,1) == nodeIndex, 2);
    
    % Find the smallest angle for this node
    minAngle = min(nodeAngles);
    
    % Store the result (node index and minimum angle)
    minAngles = [minAngles; nodeIndex, minAngle];
end
mean_branch_angle = mean(minAngles(:,2));
% % Display results
% disp('Node Index and Minimum Junction Angle (in degrees):');
% disp(minAngles);
disp(['Mean branching angle :', num2str(mean_branch_angle)]);

%% connectivity index

% Calculate the connectivity index
totalNodes = length(nodes);  % Total number of nodes
junctionNodes = 0;           % Initialize junction node count
totalLinks = length(links);   % Total number of links

% Count the number of junction nodes (nodes with more than one connection)
for i = 1:totalNodes
    if length(nodes(i).links) > 1  % Junction nodes have more than one link
        junctionNodes = junctionNodes + 1;
    end
end

% Calculate Connectivity Index
% Method 1: Ratio of junction nodes to total nodes
connectivityIndex1 = junctionNodes / totalNodes;

% Method 2: Average connections per node (links per node)
connectivityIndex2 = totalLinks / totalNodes;

% Display results
disp(['Connectivity Index (Junction Nodes / Total Nodes): ', num2str(connectivityIndex1)]);
disp(['Connectivity Index (Average Links per Node): ', num2str(connectivityIndex2)]);

%% tortuosity 
% Initialize storage for tortuosity metrics
for i = 1:length(links)
    % Define vesselX and vesselY from the link structure
    vesselX = links(i).x;
    vesselY = links(i).y;

    % Arc-over-Chord Ratio (ACR) Calculation
    % Calculate arc length (sum of distances between consecutive points)
    arcLength = sum(sqrt(diff(vesselX).^2 + diff(vesselY).^2));

    % Calculate chord length (distance between the first and last points)
    chordLength = sqrt((vesselX(end) - vesselX(1))^2 + (vesselY(end) - vesselY(1))^2);

    % Calculate Arc-over-Chord Ratio
    arcOverChordRatio = arcLength / chordLength;
    links(i).arcOverChordRatio = arcOverChordRatio;  % Store in link structure

    % Vascular Curvature Index (VCI) Calculation
    totalAngularChange = 0;

    % Loop through points to calculate angle changes between each consecutive segment
    for j = 2:length(vesselX)-1
        % Define vectors between consecutive segments
        vec1 = [vesselX(j) - vesselX(j-1), vesselY(j) - vesselY(j-1)];
        vec2 = [vesselX(j+1) - vesselX(j), vesselY(j+1) - vesselY(j)];

        % Calculate the angle between vectors
        cosTheta = dot(vec1, vec2) / (norm(vec1) * norm(vec2));
        angleChange = acosd(cosTheta);  % Convert to degrees

        % Accumulate absolute angular changes
        totalAngularChange = totalAngularChange + abs(angleChange);
    end

    % Store VCI as the cumulative angular change per link
    links(i).VCI = totalAngularChange / (length(vesselX) - 2);
end

% % Display calculated tortuosity metrics for each link
% for i = 1:length(links)
%     disp(['Link ', num2str(i), ': Arc-over-Chord Ratio = ', num2str(links(i).arcOverChordRatio)]);
%     disp(['Link ', num2str(i), ': Vascular Curvature Index (VCI) = ', num2str(links(i).VCI)]);

% Collect tortuosity values for each metric
arcOverChordRatios = [links.arcOverChordRatio];
VCIs = [links.VCI];

% Calculate mean and standard deviation for each metric
meanArcOverChord = mean(arcOverChordRatios);
stdArcOverChord = std(arcOverChordRatios);

meanVCI = mean(VCIs);
stdVCI = std(VCIs);

% Display the results
disp(['Mean Arc-over-Chord Ratio: ', num2str(meanArcOverChord)]);
disp(['Std Dev Arc-over-Chord Ratio: ', num2str(stdArcOverChord)]);
disp(['Mean Vascular Curvature Index (VCI): ', num2str(meanVCI)]);
disp(['Std Dev Vascular Curvature Index (VCI): ', num2str(stdVCI)]);

% end
