% load('FV+_HOG_LBP_SFTA_HARALICK.mat')
% fused = horzcat(FV_HoG,FV_Haralick,FV_LBP,FV_SFTA);
% load('network_new_train_inceptionv3_.mat');
% fused_inception_HC = horzcat(trainingFeatures, fused);
% 
% Final_Features=cell2table(horzcat(Labels,num2cell(fused_inception_HC)));
% 
% save('Fused_Inception_Hand_Crafted');
%%
load('Final_Features.mat')

load('network_new_train_inceptionv3_.mat')
trainingFeatures=double(trainingFeatures);

[r, c]=size(FV_HoG);
  new_score = Find_Entropy(FV_HoG,c);
  red_dim_Hog = real(new_score(:,1:4000));
[r1, c1]=size(FV_LBP);
  new_score1 = Find_Entropy(FV_LBP,c1);
  red_dim_LBP = new_score1(:,1:50);
[r2, c2]=size(trainingFeatures);
  new_score3 = Find_Entropy(trainingFeatures,c2);
  red_dim_Inception_Feature = new_score3(:,1:400);

  
fused = horzcat(red_dim_Hog,red_dim_LBP);
fused_features_entropy = horzcat(red_dim_Inception_Feature, fused);   


Final_Features_1=cell2table(horzcat(Labels,num2cell(fused_features_entropy)));
  
  
save('Final_Features_1');  
  
  
