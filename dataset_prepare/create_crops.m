clc
clear
%%

generatecrops('middleburry',1800/2,100,'train',0)
generatecrops('middleburry',1800/2,100,'test',0)

generatecrops('middleburry',1400/2,150,'train',100)
generatecrops('middleburry',1400/2,150,'test',100)

generatecrops('middleburry',1000/2,250,'train',200)
generatecrops('middleburry',1000/2,250,'test',200)


generatecrops('ibims1',480/2,70,'train',0)
generatecrops('ibims1',360/2,80,'train',100)
generatecrops('ibims1',280/2,50,'train',200)
