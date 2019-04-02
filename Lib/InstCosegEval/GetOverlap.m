function [Overlap, NumInsts] = GetOverlap(Proposals, GTInsts)
GTInstsSize = size(GTInsts);
if iscell(GTInsts)
    InstFun = @(Index)CellFun(GTInsts,Index);
    NumInsts = max(GTInstsSize);
else
    if length(GTInstsSize) == 2
        NumInsts = 1;
    else
        NumInsts = GTInstsSize(3);
    end
    InstFun = @(Index)ThreeDMatFun(GTInsts,Index);
     
end
NumProposals = size(Proposals, 3);

Overlap = zeros(NumInsts, NumProposals);
for i = 1:NumProposals
    Proposal = Proposals(:,:,i);
    for j = 1:NumInsts
        GTInst = InstFun(j);
        Overlap(j,i) = sum(Proposal(:) & GTInst(:)) / sum(Proposal(:) | GTInst(:));
    end
end
end

function InstMask = CellFun(InstList, Index)
InstMask = InstList{Index};
end
function InstMask = ThreeDMatFun(InstList, Index)
InstMask = InstList(:,:,Index);
end
function InstMask = OneDMatFun(InstList, Index)
InstMask = InstList == Index;
end