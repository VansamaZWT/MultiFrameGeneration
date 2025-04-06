import numpy as np
import os
import sys
from multiprocessing import Process
import re
import glob
#D:\zhn_dataset\ThirdPersonExampleMapMatrix.0093.txt

threadNum=4
import math
import quaternion as qtn

def Quat2Numpy(q:qtn.quaternion) -> np.array:
    assert abs(q.w)<=1e-6
    return np.array([q.x,q.y,q.z])

def Slerp(q0 : qtn.quaternion,q1 : qtn.quaternion,alpha : float,idx:int):
    '''
    alpha=0 => q=q0
    alpha=1 => q=q1
    https://krasjet.github.io/quaternion/quaternion.pdf
    '''
    #if alpha<0 or alpha>1:
    #    raise NotImplementedError
    res_dot=q0.w*q1.w+q0.x*q1.x+q0.y*q1.y+q0.z*q1.z
    if res_dot>=1-1e-8:
        print('error:',idx,q0,q1)
        return q0
    
    theta=math.acos(res_dot)
    return math.sin((1-alpha)*theta)/math.sin(theta)*q0+math.sin(alpha*theta)/math.sin(theta)*q1

def Quat2Euler(q:qtn.quaternion):#将（三个旋转）复合变换的四元数转换成（弧度制的）欧拉角
    X,Y,Z,W=q.x,q.y,q.z,q.w
    ST=W*Y+X*Z
    roll=math.atan2(2*(W*X-Y*Z),1-2*(Y*Y+X*X))
    yaw=math.atan2(2*(-W*Z+X*Y),1-2*(Y*Y+Z*Z))
    if abs(ST*2-1.0)<=1e-6:
        pitch=math.pi/2
        print(2*(W*X-Y*Z),1-2*(Y*Y+X*X))
    else:
        pitch=math.asin(ST*2)
    return roll/math.pi*180,pitch/math.pi*180,yaw/math.pi*180

def Euler2Quat(roll,pitch,yaw,isDegree):#将三个旋转表示的欧拉角转换成复合变换的四元数
    if isDegree:
        roll=roll/180*math.pi
        pitch=pitch/180*math.pi
        yaw=yaw/180*math.pi
    Zaxis=(0,0,-1)
    Yaxis=(0,1,0)
    Xaxis=(1,0,0)
    q_yaw=qtn.quaternion(math.cos(0.5*yaw),math.sin(0.5*yaw*Zaxis[0]),math.sin(0.5*yaw*Zaxis[1]),math.sin(0.5*yaw*Zaxis[2]))
    q_pitch=qtn.quaternion(math.cos(0.5*pitch),math.sin(0.5*pitch*Yaxis[0]),math.sin(0.5*pitch*Yaxis[1]),math.sin(0.5*pitch*Yaxis[2]))
    q_roll=qtn.quaternion(math.cos(0.5*roll),math.sin(0.5*roll*Xaxis[0]),math.sin(0.5*roll*Xaxis[1]),math.sin(0.5*roll*Xaxis[2]))
    q_comp=q_roll*q_pitch*q_yaw
    #qc_comp=q_comp.conjugate()#计算共轭
    return q_comp

def Quat2Axis(q:qtn.quaternion):
    uv=qtn.quaternion(0,0,0,1)
    rv=qtn.quaternion(0,0,1,0)
    fv=qtn.quaternion(0,1,0,0)
    return Quat2Numpy(q.conjugate()*uv*q),Quat2Numpy(q.conjugate()*rv*q),Quat2Numpy(q.conjugate()*fv*q)

def GetStartEndID(path):
    start = 99999
    end = 0
    for filePath in glob.glob(path+"/*"):#NOTE: Can Be Modified
        if len(filePath)<3 or filePath[-3:]!='txt':
            continue
        idx = int(filePath.split('.')[1])
        start = min(start, idx)
        end = max(end, idx)
    return start*2, end*2

def CalcMatrix(sid,eid,inPath,outPath,ScenePrefix, tsid):
    np.set_printoptions(suppress=True)
    for i in range(sid,eid):
        if i % 4 == 0  or i!=eid:
            continue
        #if i!=163:
        #    continue
        idx=str(i).zfill(4)
        prev_idx1=str((i-i%4)//2).zfill(4)
        prev_idx3=str((i+i%4)//2).zfill(4)

        # with open("{}/{}.{}.txt".format(inPath, ScenePrefix, prev_idx5),'r') as prev_file5:
        #     lines = prev_file5.readlines()
        #     matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', lines[6])
        #     assert len(matches)==3
        #     loc_1 = np.array([float(matches[0]),float(matches[1]),float(matches[2])])

        #print("{}{}.{}.txt".format(inPath,ScenePrefix,idx))
        with open("{}/{}.{}.txt".format(inPath,ScenePrefix,prev_idx1),'r') as prev_file1:
            lines=prev_file1.readlines()
            matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', lines[6])
            assert len(matches)==3
            loc1=np.array([float(matches[0]),float(matches[1]),float(matches[2])])
            matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', lines[7])
            assert len(matches)==3
            ViewRotation=np.array([float(matches[0]),float(matches[1]),float(matches[2])])
            matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', lines[1])
            #print(len(matches),matches,lines[1])
            assert len(matches)==16
            rv1=np.array([float(matches[0]),float(matches[4]),float(matches[8])])
            uv1=np.array([float(matches[1]),float(matches[5]),float(matches[9])])
            fv1=np.array([float(matches[2]),float(matches[6]),float(matches[10])])
            #print(idx,ViewLocation,ViewRotation) Debug
            q1=Euler2Quat(pitch=ViewRotation[0],yaw=ViewRotation[1],roll=ViewRotation[2],isDegree=True)
            with open("{}/{}.{}.txt".format(inPath,ScenePrefix,prev_idx3),'r') as prev_file3:
                lines=prev_file3.readlines()
                matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', lines[6])
                assert len(matches)==3
                loc0=np.array([float(matches[0]),float(matches[1]),float(matches[2])])
                matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', lines[7])
                assert len(matches)==3
                ViewRotation=np.array([float(matches[0]),float(matches[1]),float(matches[2])])
                matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', lines[1])
                assert len(matches)==16
                rv0=np.array([float(matches[0]),float(matches[4]),float(matches[8])])
                uv0=np.array([float(matches[1]),float(matches[5]),float(matches[9])])
                fv0=np.array([float(matches[2]),float(matches[6]),float(matches[10])])
                #print(idx,ViewLocation,ViewRotation) Debug
                q0=Euler2Quat(pitch=ViewRotation[0],yaw=ViewRotation[1],roll=ViewRotation[2],isDegree=True)
                alpha=(i%4)/4
                q=Slerp(q0,q1,alpha,idx)
                uv,rv,fv=Quat2Axis(q)
                loc=alpha*loc1+(1-alpha)*loc0  #TODO:
                # loc = 1.625 * loc1 - 0.75 * loc0 + 0.125 * loc_1;
                ViewMatrix=np.array([
                    [rv[0],rv[1],rv[2],-np.dot(rv,loc)],
                    [uv[0],uv[1],uv[2],-np.dot(uv,loc)],
                    [fv[0],fv[1],fv[2],-np.dot(fv,loc)],
                    [0,0,0,1],
                ]).transpose()
                
                with open("{}/{}.{}.txt".format(outPath,ScenePrefix,idx),'w') as file:
                    file.write("\nOurs Extra ViewMatrix:"+str(ViewMatrix))


def CreateAndJoinCalcProcess(inPath,outPath,ScenePrefix):
    sid,eid=GetStartEndID(inPath)
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    processList=list()
    blockSize=(eid-sid+1+threadNum-1)//threadNum
    print(eid,sid,inPath)
    for i in range(threadNum):
        processList.append(Process(target=CalcMatrix,args=(sid+i*blockSize,min(sid+(i+1)*blockSize,eid+1),inPath,outPath,ScenePrefix, sid)))
    for p in processList:
        p.start()
        p.join()


if __name__ == "__main__":
    root_path = "/disk/zjw/data/zwtdataset/"
    # sequence_names = ["RedwoodForest/train1-60fps", "RedwoodForest/train2-60fps", "RedwoodForest/test1-60fps", "MedievalDocks/train1-60fps", "MedievalDocks/train2-60fps", "MedievalDocks/test1-60fps", "Showdown/test1-60fps", "Factory/test1-60fps", "WesternTown/train1-60fps", "WesternTown/test1-60fps", "EasternVillage/train1-60fps", "EasternVillage/train2-60fps", "EasternVillage/test1-60fps"]
    sequence_names = ["EasternVillage/test1-60fps"]
    for sequence_name in sequence_names:
        FilePath = os.path.join(root_path, sequence_name)
        ScenePrefix = sequence_name.split("/")[0] + "Matrix"
        CreateAndJoinCalcProcess(FilePath,"./tmp",ScenePrefix)