function [dirVector] = getDirectionVector(x, y, junction_x, junction_y)
    % Ensure x and y are column vectors
    x = x(:);
    y = y(:);
    
    % Handle vertical line case
    if range(x) == 0
        dirVector = [0, 1]; % Vertical direction
    else
        % Linear regression on limited points
        p = polyfit(x, y, 1); % y = p(1)*x + p(2)
        dirVector = [1, p(1)]; % Slope as direction vector
    end
    
    % Normalize the direction vector
    dirVector = dirVector / norm(dirVector);
    
    % Use mean point of N points as reference
    meanX = mean(x);
    meanY = mean(y);
    vecToBranch = [meanX - junction_x, meanY - junction_y];
    
    % Check the direction and flip if necessary
    if dot(dirVector, vecToBranch) < 0
        dirVector = -dirVector;
    end
end 
% function [dirVector] = getDirectionVector(x, y, junction_x, junction_y)
%     % Ensure x and y are column vectors
%     x = x(:);
%     y = y(:);
%     
%     % Perform linear regression
%     p = polyfit((x - mean(x)) / std(x), (y - mean(y)) / std(y), 1);
%     
%     % Create a direction vector (using unit vector in direction of line)
%     x_range = max(x) - min(x);
%     if x_range == 0
%         % Vertical line
%         dirVector = [0, 1];
%     else
%         x1 = min(x);
%         x2 = x1 + x_range;
%         y1 = p(1) * x1 + p(2);
%         y2 = p(1) * x2 + p(2);
%         dirVector = [x(end) - x(1), y(end) - y(1)];
%     end
%     
%     % Normalize the vector
%     dirVector = dirVector / norm(dirVector);
%     
%     % Check the direction relative to the junction
%     % Compute a vector from the junction to the first point of the branch
%     vecToBranch = [x(1) - junction_x, y(1) - junction_y];
%     % If dirVector points toward the junction, flip it
%     if dot(dirVector, vecToBranch) < 0
%         dirVector = -dirVector;
%     end
% end