import os

data_path = './data'
folder1 = 'CHASEDB1'
folder2 = 'DRIVE'
folder3 = 'STARE'
folder4 = 'HRF'
folder5 = 'DRHAGIS'

#for CHASEDB1
# CHASEDB1_path = os.path.join(data_path, folder1)
# CHASEDB1_images = os.listdir(os.path.join(CHASEDB1_path, 'image'))
# CHASEDB1_output_1stHO = os.listdir(os.path.join(CHASEDB1_path, 'output', '1stHO'))
# CHASEDB1_output_2ndHO = os.listdir(os.path.join(CHASEDB1_path, 'output', '2ndHO'))

# #rename filename 1stHO and 2ndHO name from filename to match the image name
# for i in range(len(CHASEDB1_output_1stHO)):
#     #rename 1stHO
#     os.rename(os.path.join(CHASEDB1_path, 'output', '1stHO', CHASEDB1_output_1stHO[i]), os.path.join(CHASEDB1_path, 'output', '1stHO', CHASEDB1_images[i][:-4] + '_1stHO.png'))

#     #rename 2ndHO
# for i in range(len(CHASEDB1_output_2ndHO)):
#     os.rename(os.path.join(CHASEDB1_path, 'output', '2ndHO', CHASEDB1_output_2ndHO[i]), os.path.join(CHASEDB1_path, 'output', '2ndHO', CHASEDB1_images[i][:-4] + '_2ndHO.png'))
    





#for STARE
STARE_path = os.path.join(data_path, folder3)
STARE_images = os.listdir(os.path.join(STARE_path, 'image'))
STARE_output = os.listdir(os.path.join(STARE_path, 'output'))

print('STARE')
print('Number of images:', len(STARE_images))
print('Number of masks:', len(STARE_output))

#HRF
HRF_path = os.path.join(data_path, folder4)
HRF_images = os.listdir(os.path.join(HRF_path, 'image'))
HRF_output = os.listdir(os.path.join(HRF_path, 'output'))

print('HRF')
print('Number of images:', len(HRF_images))
print('Number of masks:', len(HRF_output))

#DRHAGIS
DRHAGIS_path = os.path.join(data_path, folder5)
DRHAGIS_images = os.listdir(os.path.join(DRHAGIS_path, 'image'))
DRHAGIS_output = os.listdir(os.path.join(DRHAGIS_path, 'output'))

print('DRHAGIS')
print('Number of images:', len(DRHAGIS_images))
print('Number of masks:', len(DRHAGIS_output))
