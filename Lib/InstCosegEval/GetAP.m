function AP = GetAP(PredMasksScores, OverlapAll, Threshold, NumInsts)
NumGTs =0;
NumImgs = length(PredMasksScores);
TotalDetectedProps=0;
for k=1:NumImgs
	TotalDetectedProps=TotalDetectedProps+numel(PredMasksScores{k});
end
Scores = zeros(TotalDetectedProps, 1);
Labels = zeros(TotalDetectedProps, 1);
AllDets=0;

for k=1:NumImgs
    
%     if(rem(k-1,100)==0)
%         fprintf('Doing : %d/%d\n', k, NumImgs);
%     end
%     
    
    %add things to diagnostic
    NumDets = numel(PredMasksScores{k});
    Scores(AllDets+1:AllDets+NumDets) = PredMasksScores{k}(:);
    NumGTs = NumGTs + NumInsts{k};
    %get all Overlaps
    Overlaps = OverlapAll{k};
    if isempty(Overlaps)
        continue
    end
    %record number of ground truth
    
    
    %compute labels using Overlaps
    Labels(AllDets+1:AllDets+NumDets) =OverlapsToLabels(Scores(AllDets+1:AllDets+NumDets), Overlaps, Threshold);
    AllDets = AllDets + NumDets;
end
AP = CalAP(Scores, Labels, NumGTs);
end

function Labels= OverlapsToLabels(Scores, Overlaps, Thresh)
NumInsts = size(Overlaps, 1);
NumDets = size(Overlaps, 2);

Covered=false(NumInsts,1);
Labels=zeros(NumDets,1);
[~, SortIndex]=sort(Scores, 'descend');


%assign
for k=1:numel(SortIndex)
    if(all(Covered))
        break;
    end
    Idx = find(~Covered);
    [Overlap, AssignID] = max(Overlaps(~Covered,SortIndex(k)));
    if(Overlap > Thresh)
        Labels(SortIndex(k))=1;
        Covered(Idx(AssignID))=true;
    end
end
end

function [ap, prec, rec] = CalAP(Scores, Labels, NumGTs)
Scores=Scores(:);
Labels=Labels(:);
[~, SortID]=sort(Scores, 'descend');
tp = Labels(SortID);
fp = 1-Labels(SortID);
tp = cumsum(tp);
fp = cumsum(fp);
prec = tp./(tp+fp);

rec = tp./NumGTs;

ap = VOCap(rec,prec);
end

function ap = VOCap(rec,prec)
mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end
