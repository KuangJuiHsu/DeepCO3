function [Proposals, Score, SelectID] = NMS(Proposals, Score, Threshold)
[~, Index] = sort(Score, 'descend');
Proposals = Proposals(:,:,Index);
Count = 0;
fprintf('NMS: %d, NumOfProposals: %d\n', Count, length(Score))

NewProposals = gpuArray(Proposals(:,:,1));
NewSelectID = gpuArray(1);
SelectID = 2:size(Proposals,3);
Proposals(:,:,1) = [];

while(1)
    % Compute IoU
    Intersection = NewProposals(:,:,end) & Proposals;
    Union = NewProposals(:,:,end) | Proposals;
    Intersection = sum(sum(Intersection, 1), 2);
    Union = sum(sum(Union, 1), 2);
    Overlap =  Intersection(:) ./ (Union(:) + eps);
    SelectIndex = Overlap < Threshold;
    
    Count = Count + 1;
    fprintf('NMS: %d, NumOfProposals: %d\n', Count, length(SelectIndex))
    SelectID = SelectID(SelectIndex);
    Proposals = Proposals(:,:,SelectIndex);    
    if isempty(Proposals)
        break;
    end
    NewProposals = cat(3, NewProposals, Proposals(:,:,1));
    NewSelectID = cat(1, NewSelectID, SelectID(1));
    Proposals(:,:,1) = [];
    SelectID(1) = [];
end
% Time = toc;
fprintf('NumOfSelectedProposals: %d\n', length(NewSelectID))
Proposals = gather(NewProposals);
SelectID = gather(NewSelectID);
Score = Score(SelectID);
% fprintf('Running Time: %6.3f sec\n', Time)
end

