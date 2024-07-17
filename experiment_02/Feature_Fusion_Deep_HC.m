load('FV+_HOG_LBP_SFTA_HARALICK.mat')
fused = horzcat(FV_HoG,FV_Haralick,FV_LBP,FV_SFTA);
load('network_new_train_inceptionv3_.mat');
fused_inception_HC = horzcat(trainingFeatures, fused);

Final_Features=cell2table(horzcat(Labels,num2cell(fused_inception_HC)));

save('Fused_Inception_Hand_Crafted');