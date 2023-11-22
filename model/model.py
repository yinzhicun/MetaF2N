import os
from .ops import *
import numpy as np
import torch
import scipy

def scope_has_variables(scope):
  return len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

class Weights(object):
    def __init__(self, scope=None, init = True, init_better_model = False):
        self.weights={}
        self.scope=scope
        self.kernel_initializer=tf.initializers.VarianceScaling()
        self.loadnet = torch.load('./weights/RealESRGAN_x4plus.pth') 
        self.build_CNN_params(num_feat=64)
        
    def build_CNN_params(self,num_feat=64):

        kernel_initializer=self.kernel_initializer
        with tf.compat.v1.variable_scope(self.scope):

            self.weights['conv_first/weight'] = tf.compat.v1.Variable(name ='conv_first/weight', initial_value = self.loadnet["params_ema"]['conv_first.weight'].data.permute(2, 3, 1 ,0).numpy(), expected_shape = [3, 3, 3, num_feat], dtype=tf.float32)
            self.weights['conv_first/bias'] = tf.compat.v1.Variable(name ='conv_first/bias', initial_value = self.loadnet["params_ema"]['conv_first.bias'].data.numpy(), expected_shape = [num_feat], dtype=tf.float32)
            
            for i1 in range(23):
                for i2 in range(3):
                    for i3 in range(5):
                        self.weights[f'body/{i1}/rdb{i2+1}/conv{i3+1}/weight'] = tf.compat.v1.Variable(name = f'body/{i1}/rdb{i2+1}/conv{i3+1}/weight', initial_value = self.loadnet["params_ema"][f'body.{i1}.rdb{i2+1}.conv{i3+1}.weight'].data.permute(2, 3, 1 ,0).numpy(), expected_shape = [3, 3, num_feat + 32 * i3, 32], dtype=tf.float32)
                        self.weights[f'body/{i1}/rdb{i2+1}/conv{i3+1}/bias'] = tf.compat.v1.Variable(name = f'body/{i1}/rdb{i2+1}/conv{i3+1}/bias', initial_value = self.loadnet["params_ema"][f'body.{i1}.rdb{i2+1}.conv{i3+1}.bias'].data.numpy(), expected_shape = [32], dtype=tf.float32)

            self.weights['conv_body/weight'] = tf.compat.v1.Variable(name ='conv_body/weight', initial_value = self.loadnet["params_ema"]['conv_body.weight'].data.permute(2, 3, 1 ,0).numpy(), expected_shape = [3, 3, num_feat, num_feat], dtype=tf.float32)
            self.weights['conv_body/bias'] = tf.compat.v1.Variable(name = 'conv_body/bias', initial_value = self.loadnet["params_ema"]['conv_body.bias'].data.numpy(), expected_shape = [num_feat], dtype=tf.float32)

            self.weights['conv_up1/weight'] = tf.compat.v1.Variable(name ='conv_up1/weight', initial_value = self.loadnet["params_ema"]['conv_up1.weight'].data.permute(2, 3, 1 ,0).numpy(), expected_shape = [3, 3, num_feat, num_feat], dtype=tf.float32)
            self.weights['conv_up1/bias'] = tf.compat.v1.Variable(name = 'conv_up1/bias', initial_value = self.loadnet["params_ema"]['conv_up1.bias'].data.numpy(), expected_shape = [num_feat], dtype=tf.float32)

            self.weights['conv_up2/weight'] = tf.compat.v1.Variable(name = 'conv_up2/weight', initial_value = self.loadnet["params_ema"]['conv_up2.weight'].data.permute(2, 3, 1 ,0).numpy(), expected_shape = [3, 3, num_feat, num_feat], dtype=tf.float32)
            self.weights['conv_up2/bias'] = tf.compat.v1.Variable(name = 'conv_up2/bias', initial_value = self.loadnet["params_ema"]['conv_up2.bias'].data.numpy(), expected_shape = [num_feat], dtype=tf.float32)

            self.weights['conv_hr/weight'] = tf.compat.v1.Variable(name = 'conv_hr/weight', initial_value = self.loadnet["params_ema"]['conv_hr.weight'].data.permute(2, 3, 1 ,0).numpy(), expected_shape = [3, 3, num_feat, num_feat], dtype=tf.float32)
            self.weights['conv_hr/bias'] = tf.compat.v1.Variable(name = 'conv_hr/bias', initial_value = self.loadnet["params_ema"]['conv_hr.bias'].data.numpy(), expected_shape = [num_feat], dtype=tf.float32)

            self.weights['conv_last/weight'] = tf.compat.v1.Variable(name = 'conv_last/weight', initial_value = self.loadnet["params_ema"]['conv_last.weight'].data.permute(2, 3, 1 ,0).numpy(), expected_shape = [3, 3, num_feat, 3], dtype=tf.float32)
            self.weights['conv_last/bias'] = tf.compat.v1.Variable(name = 'conv_last/bias', initial_value = self.loadnet["params_ema"]['conv_last.bias'].data.numpy(), expected_shape = [3], dtype=tf.float32)

       
class MODEL(object):
    def __init__(self, name):
        self.name = name

        print('Build Model {}'.format(self.name))

    def forward(self, x, param):

        self.input = x
        self.param = param
    
        self.res = conv2d(self.input, param['conv_first/weight'], param['conv_first/bias'], scope='conv_first', activation=None)
        self.head = self.res

        for i1 in range(23):
            self.head1 = self.res
            for i2 in range(3):
                self.head2 = self.res
                self.x1 = conv2d(self.res, param[f'body/{i1}/rdb{i2+1}/conv1/weight'], param[f'body/{i1}/rdb{i2+1}/conv1/bias'], scope=f'body/{i1}/rdb{i2+1}/conv1', activation='leakyReLU')
                self.x2 = conv2d(tf.concat([self.res, self.x1], axis=3), param[f'body/{i1}/rdb{i2+1}/conv2/weight'], param[f'body/{i1}/rdb{i2+1}/conv2/bias'], scope=f'body/{i1}/rdb{i2+1}/conv2', activation='leakyReLU')
                self.x3 = conv2d(tf.concat([self.res, self.x1, self.x2], axis=3), param[f'body/{i1}/rdb{i2+1}/conv3/weight'], param[f'body/{i1}/rdb{i2+1}/conv3/bias'], scope=f'body/{i1}/rdb{i2+1}/conv3', activation='leakyReLU')
                self.x4 = conv2d(tf.concat([self.res, self.x1, self.x2, self.x3], axis=3), param[f'body/{i1}/rdb{i2+1}/conv4/weight'], param[f'body/{i1}/rdb{i2+1}/conv4/bias'], scope=f'body/{i1}/rdb{i2+1}/conv4', activation='leakyReLU')
                self.res = conv2d(tf.concat([self.res, self.x1, self.x2, self.x3, self.x4], axis=3), param[f'body/{i1}/rdb{i2+1}/conv5/weight'], param[f'body/{i1}/rdb{i2+1}/conv5/bias'], scope=f'body/{i1}/rdb{i2+1}/conv5', activation=None)
                self.res = tf.add(self.res * 0.2, self.head2)
            self.res = tf.add(self.res * 0.2, self.head1)
        
        self.res = conv2d(self.res, param['conv_body/weight'], param['conv_body/bias'], scope='conv_body', activation=None)
        self.res = tf.add(self.res, self.head)

        _, h1, w1, _ = self.res.get_shape().as_list()
        self.res = tf.repeat(tf.repeat(self.res, 2, axis=1), 2, axis=2)
        self.res = conv2d(self.res, param['conv_up1/weight'], param['conv_up1/bias'], scope='conv_up1', activation='leakyReLU')
        _, h2, w2, _ = self.res.get_shape().as_list()
        self.res = tf.repeat(tf.repeat(self.res, 2, axis=1), 2, axis=2)
        self.res = conv2d(self.res, param['conv_up2/weight'], param['conv_up2/bias'], scope='conv_up2', activation='leakyReLU')
        self.res = conv2d(self.res, param['conv_hr/weight'], param['conv_hr/bias'], scope='conv_hr', activation='leakyReLU')      
        self.output = conv2d(self.res, param['conv_last/weight'], param['conv_last/bias'], scope='conv_last', activation=None) 


class MaskNet():
    def __init__(self):

        self.name = "Mask"
        self.weights = {}
        self.kernel_initializer=tf.initializers.VarianceScaling()
        self.build_CNN_params()

    def forward(self, rgb):
        
        self.input = rgb
        with tf.compat.v1.variable_scope(self.name):
            self.conv1 = conv2d(self.input, self.weights['conv1/w'], self.weights['conv1/b'], scope='conv1', activation='ReLU')
            self.head = self.conv1
            for idx in range(2,8):
                self.head = conv2d(self.head, self.weights['conv%d/w' %idx], self.weights['conv%d/b' % idx], scope='conv%d' %idx, activation='ReLU')
            self.out1 = conv2d(self.head, self.weights['conv8/w'], self.weights['conv8/b'], scope='conv8', activation=None)
            #self.output = tf.sigmoid(self.out1)
            self.output = tf.nn.relu(self.out1)
    

    def build_CNN_params(self, name=None):
    
        kernel_initializer=self.kernel_initializer
        with tf.compat.v1.variable_scope(self.name):

            self.weights['conv1/w'] = tf.compat.v1.get_variable('conv1/kernel', [3, 3, 6, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv1/b'] = tf.compat.v1.get_variable('conv1/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv2/w'] = tf.compat.v1.get_variable('conv2/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv2/b'] = tf.compat.v1.get_variable('conv2/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv3/w'] = tf.compat.v1.get_variable('conv3/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv3/b'] = tf.compat.v1.get_variable('conv3/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv4/w'] = tf.compat.v1.get_variable('conv4/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv4/b'] = tf.compat.v1.get_variable('conv4/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv5/w'] = tf.compat.v1.get_variable('conv5/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv5/b'] = tf.compat.v1.get_variable('conv5/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv6/w'] = tf.compat.v1.get_variable('conv6/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv6/b'] = tf.compat.v1.get_variable('conv6/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv7/w'] = tf.compat.v1.get_variable('conv7/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv7/b'] = tf.compat.v1.get_variable('conv7/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv8/w'] = tf.compat.v1.get_variable('conv8/kernel', [3, 3, 64, 1], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv8/b'] = tf.compat.v1.get_variable('conv8/bias',[1], dtype=tf.float32, initializer=tf.zeros_initializer())

class UNetDiscriminatorSN(object):
    def __init__(self, load_pretrain_model=True):
        
        self.init = load_pretrain_model
        self.weights = {}
        self.ops = {}
        self.scope = "Discriminator"
        if self.init:
            self.loadnet = torch.load('./weights/RealESRGAN_x4plus_netD.pth')
        else:
            self.loadnet = {"params": torch.load('./weights/Random_RealESRGAN_D.pth')}

        self.build_CNN_params()
        self.if_first_load = True

    def forward(self, rgb, update_collection=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        with tf.compat.v1.variable_scope("Discriminator"):

            self.conv0 = self.conv_layer(rgb, self.weights["conv0.weight"], 1, name="conv0", bias=self.weights["conv0.bias"])
            self.conv1 = self.conv_layer(self.conv0, self.spectral_normed_weight(self.weights["conv1.weight"], name="conv1", update_collection=update_collection), 2, name="conv1")
            self.conv2 = self.conv_layer(self.conv1, self.spectral_normed_weight(self.weights["conv2.weight"], name="conv2", update_collection=update_collection), 2, name="conv2")
            self.conv3 = self.conv_layer(self.conv2, self.spectral_normed_weight(self.weights["conv3.weight"], name="conv3", update_collection=update_collection), 2, name="conv3")

            #_, h1, w1, _ = self.conv3.get_shape().as_list()
            h1 = tf.shape(self.conv3)[1]
            w1 = tf.shape(self.conv3)[2]
            self.conv3_up = tf.image.resize(self.conv3, [2*h1, 2*w1], method="bilinear", name="conv3_up")
            #self.conv3_up = tf.compat.v1.image.resize_bilinear(self.conv3, [2*h1, 2*w1], align_corners=False, name="conv3_up")
            self.conv4 = self.conv_layer(self.conv3_up, self.spectral_normed_weight(self.weights["conv4.weight"], name="conv4", update_collection=update_collection), 1, name="conv4")
            
            self.conv4 = self.conv4 + self.conv2
            #_, h2, w2, _ = self.conv4.get_shape().as_list()
            h2 = tf.shape(self.conv4)[1]
            w2 = tf.shape(self.conv4)[2]
            self.conv4_up = tf.image.resize(self.conv4, [2*h2, 2*w2], method="bilinear", name="conv4_up")
            #self.conv4_up = tf.compat.v1.image.resize_bilinear(self.conv4, [2*h2, 2*w2], align_corners=False, name="conv4_up")
            self.conv5 = self.conv_layer(self.conv4_up, self.spectral_normed_weight(self.weights["conv5.weight"],name="conv5", update_collection=update_collection), 1, name="conv5")
            
            self.conv5 = self.conv5 + self.conv1
            #_, h3, w3, _ = self.conv5.get_shape().as_list()
            h3 = tf.shape(self.conv5)[1]
            w3 = tf.shape(self.conv5)[2]
            self.conv5_up = tf.image.resize(self.conv5, [2*h3, 2*w3], method="bilinear", name="conv5_up")
            #self.conv5_up = tf.compat.v1.image.resize_bilinear(self.conv5, [2*h3, 2*w3], align_corners=False, name="conv5_up")
            self.conv6 = self.conv_layer(self.conv5_up, self.spectral_normed_weight(self.weights["conv6.weight"], name="conv6", update_collection=update_collection), 1, name="conv6")
            
            self.conv6 = self.conv6 + self.conv0
            self.conv7 = self.conv_layer(self.conv6, self.spectral_normed_weight(self.weights["conv7.weight"], name="conv7", update_collection=update_collection), 1, name="conv7")
            self.conv8 = self.conv_layer(self.conv7, self.spectral_normed_weight(self.weights["conv8.weight"], name="conv8", update_collection=update_collection), 1, name="conv8")
            self.output = self.conv_layer(self.conv8, self.weights["conv9.weight"], 1, name="conv9", bias=self.weights["conv9.bias"], leaky=False)
           

    def conv_layer(self, bottom, filter, stride, name, padding=1 ,bias=None, leaky=True):
        with tf.compat.v1.variable_scope(name):
            res = tf.nn.conv2d(bottom, filter, [1, stride, stride, 1], padding=[[0, 0], [padding, padding], [padding, padding], [0, 0]], name="conv2d")
            if bias is not None:
                res = tf.nn.bias_add(res, bias, name="biasadd")
            if leaky:
                res = tf.nn.leaky_relu(res, 0.2)
            else:
                return res
            return res

    def build_CNN_params(self, name=None):
        
        with tf.compat.v1.variable_scope(self.scope):

            self.weights["conv0.weight"] = tf.compat.v1.Variable(name='conv0/weight', initial_value=self.loadnet["params"]["conv0.weight"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[3,3,3,64], dtype=tf.float32)
            self.weights["conv0.bias"] = tf.compat.v1.Variable(name='conv0/bias', initial_value=self.loadnet["params"]["conv0.bias"].data.numpy(), expected_shape=[64], dtype=tf.float32)
            
            self.weights["conv1.weight"] = tf.compat.v1.Variable(name='conv1/weight', initial_value=self.loadnet["params"]["conv1.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[4,4,64,128], dtype=tf.float32)
            self.weights["conv2.weight"] = tf.compat.v1.Variable(name='conv2/weight', initial_value=self.loadnet["params"]["conv2.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[4,4,128,256], dtype=tf.float32)
            self.weights["conv3.weight"] = tf.compat.v1.Variable(name='conv3/weight', initial_value=self.loadnet["params"]["conv3.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[4,4,256,512], dtype=tf.float32)
            self.weights["conv4.weight"] = tf.compat.v1.Variable(name='conv4/weight', initial_value=self.loadnet["params"]["conv4.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[3,3,512,256], dtype=tf.float32)
            self.weights["conv5.weight"] = tf.compat.v1.Variable(name='conv5/weight', initial_value=self.loadnet["params"]["conv5.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[3,3,256,128], dtype=tf.float32)
            self.weights["conv6.weight"] = tf.compat.v1.Variable(name='conv6/weight', initial_value=self.loadnet["params"]["conv6.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[3,3,128,64], dtype=tf.float32)
            self.weights["conv7.weight"] = tf.compat.v1.Variable(name='conv7/weight', initial_value=self.loadnet["params"]["conv7.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[3,3,64,64], dtype=tf.float32)
            self.weights["conv8.weight"] = tf.compat.v1.Variable(name='conv8/weight', initial_value=self.loadnet["params"]["conv8.weight_orig"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[3,3,64,64], dtype=tf.float32)
            
            self.weights["conv9.weight"] = tf.compat.v1.Variable(name='conv9/weight', initial_value=self.loadnet["params"]["conv9.weight"].data.permute(2, 3, 1 ,0).numpy(), expected_shape=[3,3,64,1], dtype=tf.float32)
            self.weights["conv9.bias"] = tf.compat.v1.Variable(name='conv9/bias', initial_value=self.loadnet["params"]["conv9.bias"].data.numpy(), expected_shape=[1], dtype=tf.float32)

            self.weights["initu1"] = tf.compat.v1.Variable(name='conv1/u', initial_value=self.loadnet["params"]["conv1.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            self.weights["initu2"] = tf.compat.v1.Variable(name='conv2/u', initial_value=self.loadnet["params"]["conv2.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            self.weights["initu3"] = tf.compat.v1.Variable(name='conv3/u', initial_value=self.loadnet["params"]["conv3.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            self.weights["initu4"] = tf.compat.v1.Variable(name='conv4/u', initial_value=self.loadnet["params"]["conv4.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            self.weights["initu5"] = tf.compat.v1.Variable(name='conv5/u', initial_value=self.loadnet["params"]["conv5.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            self.weights["initu6"] = tf.compat.v1.Variable(name='conv6/u', initial_value=self.loadnet["params"]["conv6.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            self.weights["initu7"] = tf.compat.v1.Variable(name='conv7/u', initial_value=self.loadnet["params"]["conv7.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            self.weights["initu8"] = tf.compat.v1.Variable(name='conv8/u', initial_value=self.loadnet["params"]["conv8.weight_u"].data.unsqueeze(0).numpy(), dtype=tf.float32, trainable=False)
            

    def spectral_normed_weight(self, W, name, num_iters=1, update_collection=None, with_sigma=False):
        # Usually num_iters = 1 will be enough
       
        W_shape = W.shape.as_list()
        W_reshaped = tf.reshape(W, [-1, int(W_shape[-1])])
        #u = getattr(self, f"initu{name[-1]}")
        u = self.weights[f"initu{name[-1]}"]
        
        def power_iteration(i, u_i, v_i):
            v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
            u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
            return i + 1, u_ip1, v_ip1

        _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                    u, tf.zeros(dtype=tf.float32, shape=[1, int(W_reshaped.shape.as_list()[0])]))
        )
        u_final = tf.stop_gradient(u_final)
        v_final = tf.stop_gradient(v_final)

        if update_collection is None:
            sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
            W_bar = W_reshaped / sigma
            #setattr(self, f"initu{name[-1]}",u_final)
            W_bar = tf.reshape(W_bar, W_shape)
        else:
            sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
            W_bar = W_reshaped / sigma
            W_bar = tf.reshape(W_bar, W_shape)
            self.ops[f"initu{name[-1]}"] = tf.compat.v1.assign(self.weights[f"initu{name[-1]}"], u_final)
            
        if with_sigma:
            return W_bar, sigma
        else:
            return W_bar
