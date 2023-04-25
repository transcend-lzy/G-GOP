import tensorflow as tf
from tensorflow import keras

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

#gpu分配
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8) 
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,\
                                                           allow_soft_placement=True, log_device_placement=True))

import cv2
import numpy as np
import json
import moviepy.editor as mvp

import os

from math import pi,cos,sin

from PIL import Image
from matplotlib import pyplot as plt

import time

main_path = "./"
sample_path = os.path.join(main_path,'sample') #模型输出
data_path = os.path.join(main_path,'data') #训练数据

if not os.path.exists(main_path):
  os.mkdir(main_path)
if not os.path.exists(sample_path):
  os.mkdir(sample_path)
if not os.path.exists(data_path):
  os.mkdir(data_path)

def im_save(im,index,path=sample_path):  #保存图像

  if np.max(im)<=1:
    im = im*255
  im = im.astype(np.uint8)

  if np.shape(im)[-1] !=3:
    sample_im = Image.new('L', (128*8, 128*8))

    for i,x in enumerate(range(0, 128*8, 128)):
      for j,y in enumerate(range(0, 128*8, 128)):

        I1 = im[i*8+j,:,:]
        I1 = Image.fromarray(I1, mode='L')
        sample_im.paste(I1, (x, y))

    sample_im.save( os.path.join(path,'sample{}.png'.format(index)) )
  else:
    cv2.imwrite(os.path.join(path,'sample{}.png'.format(index)),im)

def show_example_im_double(im,im2):  #为了生成 输入图像和生成图像的对比图
    
  if np.max(im)<=1:
    im = im*255
  if np.max(im2)<=1:
    im2 = im2*255
    
  im = im.astype(np.uint8)
  im2 = im2.astype(np.uint8)

  new_im = Image.new('L', (128*8*2, 128*8))

  for i,x in enumerate(range(0, 128*8*2, 128*2)):
    for j,y in enumerate(range(0, 128*8, 128)):

      I = im[i*8+j,:,:]
      I[-5:-1,:]=255
      I = Image.fromarray(I, mode='L')
      new_im.paste(I, (x, y))

  for i,x in enumerate(range(0, 128*8*2, 128*2)):
    for j,y in enumerate(range(0, 128*8, 128)):

      I = im2[i*8+j,:,:]
      I[-5:-1,:]=255
      I[:,-5:-1]=255
      I = Image.fromarray(I, mode='L')
      new_im.paste(I, (x+128, y))

  fig= plt.figure(figsize=(12,6),dpi=128*8/6.)
  plt.imshow(np.asarray(new_im),cmap='gray')
  plt.axis('off')
  plt.show()
class CVAE(tf.keras.Model):
  def __init__(self, latent_dim, dim=32*5):  #latent_dim8维 (x,y)和6自由度
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.dim = dim

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=100, activation=tf.nn.relu),  #100维全连接层(让他有更多变化)
          tf.keras.layers.Dense(units=100, activation=tf.nn.relu),  #100维全连接层
#           tf.keras.layers.Dense(units=4*4*self.dim*8, activation=tf.nn.relu),
          tf.keras.layers.Dense(units=4*4*self.dim*16, activation=tf.nn.relu), 
          tf.keras.layers.Reshape(target_shape=(4, 4, self.dim*16)),#dense和reshape构成一个卷积层,得到一个立方体
          keras.layers.BatchNormalization(),  #bn层
             
             
            
          
        # 4x4x(dim*16) => 8x8x(dim*8) 
          tf.keras.layers.Conv2DTranspose(
                  filters=self.dim*8, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),#反卷积
         
            keras.layers.BatchNormalization(),
         
            # 8x8x(dim*8) => 16x16x(dim*4) 
          tf.keras.layers.Conv2DTranspose(
              filters=self.dim*4, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
          keras.layers.BatchNormalization(),
          # 16x16x(dim*4) => 32x32x(dim*2) 
          tf.keras.layers.Conv2DTranspose(
              filters=self.dim*2, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
          keras.layers.BatchNormalization(),
          # 32x32x(dim*2) => 64x64xdim 
          tf.keras.layers.Conv2DTranspose(
              filters=self.dim, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
          keras.layers.BatchNormalization(),
          # 64x64xdim => 128x128x1 (No activation)
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=5, strides=(2, 2), padding="SAME"),
        ]
    )

  @tf.function
  def decode(self, z, apply_sigmoid=False):  #做了一个sigmoid  (变成0-1)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)  #logits是哪里定义的
      return probs

    return logits

latent_dim = 8
dim=32*7  #实验得出
num_examples_to_generate = 64  #得到样本数

model = CVAE(latent_dim,dim)

@tf.function
def compute_loss(model, x, z):  #输入z是8维输入特征,x是训练图,输出由z经model的生成图像和x之间的loss
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  rec_loss = tf.reduce_sum(cross_ent) / x.shape[0]

  return tf.reduce_mean(rec_loss)

@tf.function
def compute_apply_gradients(model, x, z, optimizer):#优化器
  with tf.GradientTape() as tape:
    rec_loss = compute_loss(model, x, z)
  gradients = tape.gradient(rec_loss, model.trainable_variables)
  gradients,_ = tf.clip_by_global_norm(gradients, 15)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return rec_loss 

def restore_checkpoint(manager):  #恢复预训练模型
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

def save_checkpoint(manager):  #存储预训练模型
  ckpt.step.assign_add(1)  
  if int(ckpt.step) %500 == 0:
    save_path = manager.save()
    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

optimizer = tf.keras.optimizers.Adam(2e-4)
#用来做持久化，把函数的中间变量都保存起来。
checkpoint_dir = os.path.join(main_path,'ckpt')
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)  #确定要存哪些参数，形参就是想保存的值
#Tensorflow的Checkpoint机制将可追踪变量以二进制的方式储存成一个.ckpt文件，储存了变量的名称及对应张量的值。
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)  #max_to_keep 保留3个存档

#读档
restore_checkpoint(manager)
start_index = int(ckpt.step) 
print("start index: {}".format(start_index))

#读取训练数据
pose_set = np.load('./pose_set/0-10000.npy')  #八维向量

for i in range(1,32):  #32 原来是128
    name = './pose_set/{}-{}.npy'.format(i*10000,(i+1)*10000)
    pose = np.load(name)
    pose_set = np.concatenate((pose_set,pose))
    
pose_set_v = np.load('./pose_set/validation(640).npy')  #验证集

print(np.shape(pose_set))
print(np.shape(pose_set_v))


train_data_name = ['dataset_uint8_1.npy','dataset_uint8_2.npy','dataset_uint8_3.npy','dataset_uint8_4.npy',
                   'dataset_uint8_5.npy','dataset_uint8_6.npy','dataset_uint8_7.npy','dataset_uint8_8.npy']

train_data = np.load(train_data_name[0])
print(np.shape(train_data))

valid_data = np.load('dataset_uint8(validation640).npy')
print(np.shape(valid_data))

im = valid_data[:64,:,:]
i=499
im2 = train_data[i*64:i*64+64,:,:]
show_example_im_double(im,im2)



#训练代码
test_z_loss_path = os.path.join(data_path,'test_z_loss')
train_z_loss_path = os.path.join(data_path,'train_z_loss')

if not os.path.exists(test_z_loss_path):
  os.mkdir(test_z_loss_path)
if not os.path.exists(train_z_loss_path):
  os.mkdir(train_z_loss_path)

im_test = valid_data[:64,:,:]  #验证图像
im_test = im_test.astype(np.float32)/255.
test_x = tf.reshape(im_test, [-1, 128, 128, 1])  #test_x是验证集图像
test_p = pose_set_v[:64,:]   #test_p是验证集8维向量
test_p = np.concatenate((test_p[:,:3],test_p[:,5:]),1)
test_p = test_p.astype(np.float64)

train_loss_data = np.zeros([100,1])
test_loss_data = np.zeros(1)

start_time = time.time()

batch_index=0
dataset_index = 0

for i in range(start_index,10000001):
    pose_index = i%20000
    pose = pose_set[pose_index*64:(pose_index+1)*64,:]  #64张图片一组
    z = np.concatenate((pose[:,:3],pose[:,5:]),1)  # 012  56789,10 一共8维
    z = z.astype(np.float64)  #z是训练图对应的姿态,八维向量
    
    check = (i//1250)%16  #一个数据集8w照片够训练1250次，一共有16套图片
    if check != dataset_index:
        train_data = np.load('./dataset_uint8_{}.npy'.format(check+1))
        dataset_index = check
    batch_index = i%1250  #80000张图像64张一组，一共1250组
    
    batch_start_time = time.time()

    im = train_data[batch_index*64:batch_index*64+64,:,:]
    
    im = im.astype(np.float32)/255.  
    
    train_x = tf.reshape(im, [-1, 128, 128, 1])#train_x是训练图像
    
    batch_end_time = time.time()
    rec_loss = compute_apply_gradients(model, train_x, z, optimizer)  #训练
    
    a='{}'.format(rec_loss)
     
    assert a !='nan'
    
    batch_time = batch_end_time-batch_start_time
    all_process_time = int(batch_end_time-start_time)
    all_process_time_h = all_process_time//60**2
    all_process_time_m = (all_process_time- all_process_time_h*60**2)//60
    all_process_time_s = all_process_time- all_process_time_h*60**2-all_process_time_m*60
    
    print('iteration:{}  rec_loss:{:4f} time: {:.2f}s /{}:{}:{} '\
          .format(i,rec_loss,\
            batch_time,all_process_time_h,all_process_time_m,all_process_time_s))

    train_loss_data[i%100] = rec_loss

    if i%100==0:  #100个epoch用验证集测试一下
      np.save(os.path.join(train_z_loss_path,"train_loss_{}-{}".format(i-100,i)),train_loss_data)

      rec_loss = compute_loss(model,test_x,test_p)
      print('Test Sample: iteration:{}  rec_loss:{:4f}'.format(i,rec_loss))
      test_loss_data[0] = rec_loss
      np.save(os.path.join(test_z_loss_path,"test_loss_{}".format(i)),test_loss_data)

        #把八维向量扔进去能生成图像
      x_decode = model.decode(test_p,True)*255  #model.decode因为sigmoid是0-1之间
      x_decode = x_decode.numpy()
      x_decode = np.reshape(x_decode,[-1,128,128])  #最终图像
      im_save(x_decode.astype(np.uint8),i)
      show_example_im_double(im_test,x_decode)
        

    save_checkpoint(manager)