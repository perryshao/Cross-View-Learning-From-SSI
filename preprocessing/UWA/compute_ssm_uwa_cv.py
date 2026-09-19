'''Generate the SSI for UWA3D dataset'''
import numpy as np
import h5py


def load_data(filepath,scale):
    file1 = h5py.File(filepath+'Train_Raw_cv'+scale+'.h5','r')
    x_train = file1['x_train'][:]
    y_train = file1['y_train'][:]
    file1.close()
    file2 = h5py.File(filepath+'Test_Raw_cv'+scale+'.h5','r')
    x_test = file2['x_test'][:]
    y_test = file2['y_test'][:]
    file2.close()

    return [x_train, y_train, x_test, y_test]
    
def calculate_ssm(filepath,scale):
    x_train, y_train, _, _ = load_data(filepath,scale)
    x_train_ssm = np.zeros([x_train.shape[0],x_train.shape[1],x_train.shape[2]/3,x_train.shape[2]/3,1],dtype="float32")
    sample_n = 0
    for sample in range(x_train.shape[0]):
        if np.count_nonzero(x_train[sample]) == 0:
            continue
        print("Generating %dth ssm at scale %s for trainset" % (sample,scale))
        SSM=np.zeros([x_train.shape[2]/3,x_train.shape[2]/3],dtype="float32")
        for t in range(x_train.shape[1]):
            for j1 in range(x_train.shape[-1]/3):
                current_joint = x_train[sample,t,j1*3:(j1+1)*3]
                for j2 in range(j1+1,x_train.shape[-1]/3):
                    rest_joint = x_train[sample,t,j2*3:(j2+1)*3]
                    SSM[j1,j2] = np.sqrt(np.sum(np.square(current_joint-rest_joint)))
            SSM += SSM.T # copy upper tri to lower tri
            # SSM = (SSM - np.min(SSM))/(np.max(SSM) - np.min(SSM)+1e-8) # normalize the images with same scales
            SSM = (SSM-np.mean(SSM))/(np.std(SSM)+1e-8) #normalize the images with zero mean and unit variation
            x_train_ssm[sample_n,t,:,:,0] = SSM
        x_train_ssm[sample_n,:,:,:,0] = (x_train_ssm[sample_n,:,:,:,0] - np.min(x_train_ssm[sample_n,:,:,:,0]))/(np.max(x_train_ssm[sample_n,:,:,:,0]) - np.min(x_train_ssm[sample_n,:,:,:,0])+1e-8)
        sample_n += 1
    x_train_ssm = np.delete(x_train_ssm,np.s_[sample_n:],axis=0)
    y_train = np.delete(y_train,np.s_[sample_n:],axis=0)
    file1 = h5py.File(filepath+'Trainset'+scale+'.h5','w')
    file1.create_dataset('X_train', data = x_train_ssm)
    file1.create_dataset('Y_train', data = y_train)
    file1.close()
    del x_train_ssm
    del x_train
    del y_train
    del SSM
    
    _, _, x_test, y_test = load_data(filepath,scale)
    x_test_ssm = np.zeros([x_test.shape[0],x_test.shape[1],x_test.shape[2]/3,x_test.shape[2]/3,1],dtype="float32")
    sample_n = 0
    for sample in range(x_test.shape[0]):
        if np.count_nonzero(x_test[sample]) == 0:
            continue
        print("Generating %dth ssm at scale %s for testset" % (sample,scale))
        SSM=np.zeros([x_test.shape[2]/3,x_test.shape[2]/3],dtype="float32")
        for t in range(x_test.shape[1]):
            for j1 in range(x_test.shape[-1]/3):
                current_joint = x_test[sample,t,j1*3:(j1+1)*3]
                for j2 in range(j1+1,x_test.shape[-1]/3):
                    rest_joint = x_test[sample,t,j2*3:(j2+1)*3]
                    SSM[j1,j2] = np.sqrt(np.sum(np.square(current_joint-rest_joint)))
            SSM += SSM.T # copy upper tri to lower tri
            # SSM = (SSM - np.min(SSM))/(np.max(SSM) - np.min(SSM)+1e-8) # normalize the images with same scales
            # dst = plt.imshow(SSM)
            # plt.show()
            SSM = (SSM-np.mean(SSM))/(np.std(SSM)+1e-8) #normalize the images with zero mean and unit variation
            x_test_ssm[sample_n,t,:,:,0] = SSM
        x_test_ssm[sample_n,:,:,:,0] = (x_test_ssm[sample_n,:,:,:,0] - np.min(x_test_ssm[sample_n,:,:,:,0]))/(np.max(x_test_ssm[sample_n,:,:,:,0]) - np.min(x_test_ssm[sample_n,:,:,:,0])+1e-8)
        sample_n += 1
    x_test_ssm = np.delete(x_test_ssm,np.s_[sample_n:],axis=0)
    y_test = np.delete(y_test,np.s_[sample_n:],axis=0)
    file2 = h5py.File(filepath+'Testset'+scale+'.h5','w')
    file2.create_dataset('X_test', data = x_test_ssm)
    file2.create_dataset('Y_test', data = y_test)
    file2.close()

if __name__=='__main__':
    filepath = '/home/data/UWA3DII/'
    joint_num = 15
    
    print('computing ssm images....')
    calculate_ssm(filepath,scale ='1') 
    calculate_ssm(filepath,scale ='2') 
    calculate_ssm(filepath,scale ='3')
	
  
