function [x_points, y_points] = getFirstNPoints(link, junction_x, junction_y, N)
    % Determine which end of the link is closer to the junction
    start_dist = pdist2([junction_x, junction_y], [link.x(1), link.y(1)]);
    end_dist = pdist2([junction_x, junction_y], [link.x(end), link.y(end)]);
    
    % Extract the first N points starting from the closer end
    if start_dist < end_dist
        x_points = link.x(1:min(N, length(link.x)));  % First N points from start
        y_points = link.y(1:min(N, length(link.y)));
    else
        x_points = link.x(end:-1:max(end-N+1, 1));    % Last N points reversed
        y_points = link.y(end:-1:max(end-N+1, 1));
    end
end
