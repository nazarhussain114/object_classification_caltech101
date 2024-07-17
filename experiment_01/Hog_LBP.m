clc
clear 
pathname='D:\Caltech101\25_ObjectCategories';
imagespath=imageSet(pathname,'recursive');
imagecount=1;
allfoldernames= struct2table(dir(pathname));

for (i=3:height(allfoldernames))
    new(i-2)=allfoldernames.name(i);
end

%minClassCount = min([imagespath.Count]);
%imagespath = partition(imagespath, minClassCount, 'randomize'); % Or 'sequential'


%trainingDataSizePercent = 50;
%[trainingSet, validationSet] = partition(imagespath, trainingDataSizePercent/100, 'sequential');%
      
for i=1 : size(trainingSet,2)
    m=size(trainingSet(i).ImageLocation,2);
    temp=trainingSet(i).ImageLocation;
        for j=1 :  m
            v{imagecount,1}=temp{j};
            if(~isempty(strfind(temp{j},new(1,i))))
                v{imagecount,2}=new(1,i);    
            else
                v{imagecount,2}='None';
            end
%           if(~isempty(strfind(temp{j},'accordion')))
%               v{imagecount,2}='accordin';
%           else
%               v{imagecount,2}='o';
%           end         
            img=imread(v{imagecount,1}); 
            if(size(img, 3) == 3)
            img=double(rgb2gray(img));
            end
             img=imresize(img,[128,128]);
             [featureVectorHoG, hogVisualization] = extractHOGFeatures(img);
            featureVectorLBP = extractLBPFeatures(img);
            
             Feature_HoG{imagecount,1}=featureVectorHoG(1,:);   
            Feature_LBP{imagecount,1}=featureVectorLBP(1,:);
            
            
             imagecount=imagecount+1;
     end
end
%%Creating Feature Vector
for i=1:length(Feature_HoG)
    ftemp=double(Feature_HoG{i});
    FV_HoG(i,:)=ftemp;
end
for i=1:length(Feature_LBP)
    ftemp1=double(Feature_LBP{i});
    FV_LBP(i,:)=ftemp1;
end

Labels=v(:,2);

% % F_HoG=cell2table(horzcat(Labels,num2cell(FV_HoG)));
% F_LBP=cell2table(horzcat(Labels,num2cell( FV_LBP)));
% 
% save('FV_LBP')           
            