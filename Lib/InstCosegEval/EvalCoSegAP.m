function [APAll, Overlap, NumInsts] = EvalCoSegAP(PredMasks, PredMasksScores, GTInstMasks, Threshold)
% PredMasks: the Masks from the algorthm
% PredMasks: the Mask scores from the algorthm
% GTInstMasks: the ground thruth of the instances
% APAll: the average precision at thresholds in Threshold
NumImgs = length(PredMasks);
Overlap = cell(1, NumImgs);
NumInsts = cell(1, NumImgs);
for i=1:NumImgs
    if(rem(i-1,10)==0) 
        fprintf('Computing overlaps:%d/%d\n',i, NumImgs); 
    end
    [Overlap{i}, NumInsts{i}] = GetOverlap(PredMasks{i}, GTInstMasks{i});
end 

% Count = 0;
% for i=1:NumImgs
%     [~, Index] = sort(Overlap{i}, 2,'descend');
%     if length(Index) == length(unique(Index))
%         Overlap{i} = Overlap{i}(:,Index(:,1));
%         PredMasksScores{i} = PredMasksScores{i}(Index(:,1));
%     else
%         Count = Count + 1;
%     end
% end 

APAll = zeros(length(Threshold), 1);
for t=1:length(Threshold)
    APAll(t) = GetAP(PredMasksScores, Overlap, Threshold(t), NumInsts);
end


end


