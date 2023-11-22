from utils.utils import *
from lpipstf import lpips_tf
from model import model
import time

class Train(object):
    def __init__(self, trial, step, size, meta_batch_size, meta_lr, meta_iter, task_batch_size, task_lr, task_iter, data_generator, checkpoint_dir, conf):
        print('[*] Initialize Training')
        
        self.trial = trial
        self.step = step

        self.HEIGHT = size[0]
        self.WIDTH = size[1]
        self.CHANNEL = size[2]
        self.HEIGHT1 = size[3]
        self.WIDTH1 = size[4]

        self.META_BATCH_SIZE = meta_batch_size
        self.META_LR = meta_lr
        self.META_ITER = meta_iter

        self.TASK_BATCH_SIZE = task_batch_size
        self.TASK_LR = task_lr
        self.TASK_ITER = task_iter

        self.patch_size = 64
        self.data_generator = data_generator
        self.checkpoint_dir = checkpoint_dir
        self.conf = conf

        '''placeholders'''
        self.inputa = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT//4, self.WIDTH//4, self.CHANNEL])
        self.inputb = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT1//4, self.WIDTH1//4, self.CHANNEL])
        
        self.labela = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.labelb = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT1, self.WIDTH1, self.CHANNEL])
        self.labelb_nousm = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT1, self.WIDTH1, self.CHANNEL])

        '''model'''
        self.PARAM = model.Weights(scope='MODEL')
        self.weights = self.PARAM.weights
        self.MODEL = model.MODEL(name='MODEL')
        self.discrimator = model.UNetDiscriminatorSN()
        self.mask_net = model.MaskNet()
        #self.ops = []

    def construct_model(self):
        self.stop_grad=tf.Variable(True, name='stop_grad', trainable=False)

        def task_metalearn(inp):

            inputa, inputb, labela, labelb, labelb_nousm = inp
            loss_func = tf.losses.absolute_difference
            loss_func_gan = tf.nn.sigmoid_cross_entropy_with_logits
            task_outputbs, task_lossesb, d_losses = [], [], []
            task_lossesb_vgg, task_lossesb_l1, task_lossesb_gan = [], [], []
            ops = []
    
            self.MODEL.forward(inputa, self.weights)
            task_outputa = self.MODEL.output

            self.MODEL.forward(inputb, self.weights)
            task_outputb = self.MODEL.output
            
            self.discrimator.forward(tf.stop_gradient(task_outputb), update_collection="spectral_norm_update_ops")
            task_fake_g_pred = self.discrimator.output
            
            for ops_ in self.discrimator.ops.values():
                ops.append(ops_)
            task_outputbs.append(task_outputb)
            weights = self.MODEL.param

            real = tf.constant(np.ones(task_fake_g_pred.get_shape().as_list()), dtype=tf.float32)
            fake = tf.constant(np.zeros(task_fake_g_pred.get_shape().as_list()), dtype=tf.float32)
            
            inputa_resize = tf.compat.v1.image.resize_bicubic(inputa, [self.HEIGHT, self.WIDTH], align_corners=False, name="resize_input")
            self.mask_net.forward(tf.concat([inputa_resize, labela], axis=3))
            W = self.mask_net.output

            task_lossa = tf.reduce_mean(W * tf.abs(labela - task_outputa))
            task_lossb = tf.reduce_mean(tf.abs(labelb - task_outputb)) + 0.5 * tf.reduce_mean(lpips_tf.lpips(labelb, task_outputb, model='net-lin', net='alex')) + 0.1 * tf.reduce_mean(loss_func_gan(logits=task_fake_g_pred, labels=real)) + 0.002 * tf.reduce_mean((W-1)**2) / 2.0
            
            task_lossesb_l1.append(tf.reduce_mean(tf.abs(labelb - task_outputb)))
            task_lossesb_vgg.append(tf.reduce_mean(lpips_tf.lpips(labelb, task_outputb, model='net-lin', net='alex')))
            task_lossesb_gan.append(tf.reduce_mean(loss_func_gan(logits=task_fake_g_pred, labels=real)))
            

            grads = tf.gradients(task_lossa, list(weights.values()))
            grads = tf.cond(self.stop_grad, lambda: [tf.stop_gradient(grad) for grad in grads], lambda: grads)

            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(
                zip(weights.keys(), [weights[key] - self.TASK_LR * gradients[key] for key in weights.keys()]))

            self.MODEL.forward(inputb, fast_weights)
            output = self.MODEL.output

            self.discrimator.forward(output, update_collection="spectral_norm_update_ops")
            task_fake_g_pred_ = self.discrimator.output

            task_outputbs.append(output)
            
            task_lossesb.append(tf.reduce_mean(tf.abs(labelb - output)) + 0.5 * tf.reduce_mean(lpips_tf.lpips(labelb, output, model='net-lin', net='alex')) + 0.1 * tf.reduce_mean(loss_func_gan(logits=task_fake_g_pred_, labels=real)) + 0.002 * tf.reduce_mean((W-1)**2) / 2.0)
            task_lossesb_l1.append(tf.reduce_mean(tf.abs(labelb - output)))
            task_lossesb_vgg.append(tf.reduce_mean(lpips_tf.lpips(labelb, output, model='net-lin', net='alex')))
            task_lossesb_gan.append(tf.reduce_mean(loss_func_gan(logits=task_fake_g_pred_, labels=real)))

            self.discrimator.forward(labelb_nousm, update_collection="spectral_norm_update_ops")
            real_d_pred = self.discrimator.output
            self.discrimator.forward(tf.stop_gradient(output), update_collection="spectral_norm_update_ops")
            fake_d_pred = self.discrimator.output
            d_losses.append(tf.reduce_mean(loss_func_gan(logits=real_d_pred, labels=real) + loss_func_gan(logits= fake_d_pred, labels=fake)))
            # d_losses.append(loss_func_gan(logits=task_real_g_pred_, labels=real) + loss_func_gan(logits= task_fake_g_pred_, labels=fake))
           
            for j in range(self.TASK_ITER - 1):

                self.MODEL.forward(inputa, fast_weights)
                output_s = self.MODEL.output

                loss = tf.reduce_mean(W * tf.abs(labela - output_s))
                grads = tf.gradients(loss, list(fast_weights.values()))
                grads = tf.cond(self.stop_grad, lambda: [tf.stop_gradient(grad) for grad in grads], lambda: grads)

                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.TASK_LR * gradients[key] for key in fast_weights.keys()]))

                self.MODEL.forward(inputb, fast_weights)
                output=self.MODEL.output

                self.discrimator.forward(output, update_collection="spectral_norm_update_ops")
                fake_g_pred = self.discrimator.output

                task_outputbs.append(output)
                task_lossesb.append(tf.reduce_mean(tf.abs(labelb - output)) + 0.5 * tf.reduce_mean(lpips_tf.lpips(labelb, output, model='net-lin', net='alex')) + 0.1 * tf.reduce_mean(loss_func_gan(logits=fake_g_pred, labels=real)) + 0.002 * tf.reduce_mean((W-1)**2) / 2.0)
                task_lossesb_l1.append(tf.reduce_mean(tf.abs(labelb - output)))
                task_lossesb_vgg.append(tf.reduce_mean(lpips_tf.lpips(labelb, output, model='net-lin', net='alex')))
                task_lossesb_gan.append(tf.reduce_mean(loss_func_gan(logits=fake_g_pred, labels=real)))
                
                self.discrimator.forward(labelb_nousm, update_collection="spectral_norm_update_ops")
                real_d_pred = self.discrimator.output
                self.discrimator.forward(tf.stop_gradient(output), update_collection="spectral_norm_update_ops")
                fake_d_pred = self.discrimator.output
                d_losses.append(tf.reduce_mean(loss_func_gan(logits=real_d_pred, labels=real) + loss_func_gan(logits=fake_d_pred, labels=fake)))
              
            task_output = [task_outputa, task_outputbs,  task_lossa, task_lossb, task_lossesb, d_losses, task_lossesb_l1, task_lossesb_vgg, task_lossesb_gan, ops, W]
            return task_output

        out_dtype = [tf.float32, [tf.float32] * (self.TASK_ITER + 1), tf.float32, tf.float32, [tf.float32] * self.TASK_ITER, [tf.float32] * self.TASK_ITER, [tf.float32] * (self.TASK_ITER + 1), [tf.float32] * (self.TASK_ITER + 1), [tf.float32] * (self.TASK_ITER + 1), [tf.float32] * 8, tf.float32]
        result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb, self.labelb_nousm), dtype=out_dtype,
                           parallel_iterations=self.META_BATCH_SIZE)

        self.outputas, self.outputbs, self.lossesa, self.lossb, self.lossesb, self.d_losses, self.lossesb_l1, self.lossesb_vgg, self.lossesb_gan, self.update_ops, self.W = result
        
    def __call__(self):

        PRINT_ITER = 20
        SAVE_ITER = 1000
        SECOND_ORDER_GRAD_ITER = 0 # For the 1st-order approximation. Until this step, 1st-order approximation is used for fast training

        print('[*] Setting Train Configuration')

        self.construct_model()
        
        self.global_step = tf.Variable(self.step, name='global_step', trainable=False)
        self.second_grad_on = tf.assign(self.stop_grad, False)
        self.add_step = tf.assign_add(self.global_step, 1)

        '''losses'''
        self.total_loss1 = tf.reduce_sum(self.lossesa) / tf.to_float(self.META_BATCH_SIZE)
        self.total_loss2 =  tf.reduce_sum(self.lossb) / tf.to_float(self.META_BATCH_SIZE)
        self.total_losses2 =  [tf.reduce_sum(self.lossesb[j]) / tf.to_float(self.META_BATCH_SIZE) for j in range(self.TASK_ITER)]
        self.total_d_loss =  [tf.reduce_sum(self.d_losses[j]) / tf.to_float(self.META_BATCH_SIZE) for j in range(self.TASK_ITER)]

        self.total_losses2_l1 =  [tf.reduce_sum(self.lossesb_l1[j]) / tf.to_float(self.META_BATCH_SIZE) for j in range(self.TASK_ITER+1)]
        self.total_losses2_vgg =  [tf.reduce_sum(self.lossesb_vgg[j]) / tf.to_float(self.META_BATCH_SIZE) for j in range(self.TASK_ITER+1)]
        self.total_losses2_gan =  [tf.reduce_sum(self.lossesb_gan[j]) / tf.to_float(self.META_BATCH_SIZE) for j in range(self.TASK_ITER+1)]
   
        '''weighted loss'''
        self.LW=self.get_loss_weights()
        self.weighted_total_losses2 = tf.reduce_mean(tf.multiply(tf.convert_to_tensor(self.total_losses2), self.LW))
        #self.weighted_total_losses_d = tf.reduce_mean(tf.multiply(tf.convert_to_tensor(self.total_d_loss),self.LW))
        self.weighted_total_losses_d = self.total_d_loss[-1]

        '''Optimizers'''
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MODEL|Mask')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        self.opt = tf.train.AdamOptimizer(self.META_LR)
        self.gvs = self.opt.compute_gradients(self.weighted_total_losses2, g_vars)
        self.metatrain_op= self.opt.apply_gradients(self.gvs)

        self.d_opt = tf.train.AdamOptimizer(1e-4)
        self.gds = self.d_opt.compute_gradients(self.weighted_total_losses_d, d_vars)
        self.d_op= self.d_opt.apply_gradients(self.gds)

        # self.update_ops = tf.get_collection("spectral_norm_update_ops")
        # print(self.update_ops)

        '''Summary'''
        self.summary_op = tf.summary.merge([tf.summary.scalar('Train inner_loop loss', self.total_loss1)]+
                                           [tf.summary.scalar('Train outer_loop loss, step 0', self.total_loss2)]+
                                           [tf.summary.scalar('Train outer_loop loss, step %d' % (j+1), self.total_losses2[j]) for j in range(self.TASK_ITER)]+
                                           [tf.summary.scalar('Discriminator loss, step %d' % (j+1), self.total_d_loss[j]) for j in range(self.TASK_ITER)]+
                                           [tf.summary.scalar('Train outer_loop l1_loss, step %d' % (j), self.total_losses2_l1[j]) for j in range(self.TASK_ITER+1)]+
                                           [tf.summary.scalar('Train outer_loop vgg_loss, step %d' % (j), self.total_losses2_vgg[j]) for j in range(self.TASK_ITER+1)]+
                                           [tf.summary.scalar('Train outer_loop gan_loss, step %d' % (j), self.total_losses2_gan[j]) for j in range(self.TASK_ITER+1)]+
                                           [tf.summary.image('1.inputa_query', tf.clip_by_value(self.inputa[0], 0., 1.),
                                                             max_outputs=4),
                                            tf.summary.image('2.labela_query', tf.clip_by_value(self.labela[0], 0., 1.),
                                                             max_outputs=4),
                                            tf.summary.image('3.inputb_query', tf.clip_by_value(self.inputb[0], 0., 1.),
                                                             max_outputs=4),
                                            tf.summary.image('4.init_outputb_query', tf.clip_by_value(self.outputbs[0][0], 0., 1.),
                                                             max_outputs=4),
                                            tf.summary.image('5.outputb_query', tf.clip_by_value(self.outputbs[self.TASK_ITER][0], 0., 1.),
                                                             max_outputs=4),
                                            tf.summary.image('6.GT', self.labelb[0], max_outputs=4),
                                            tf.summary.image('7.GT_Nousm', self.labelb_nousm[0], max_outputs=4),
                                            tf.summary.image('8.W', self.W[0], max_outputs=4),
                                        ])


        self.saver = tf.train.Saver(max_to_keep=100000)
        self.init=tf.global_variables_initializer()

        count_param(scope='MODEL|Mask')

        with tf.Session(config=self.conf) as sess:
            sess.run(self.init)
      
            could_load, model_step = load(self.saver, sess, self.checkpoint_dir, folder='Model%d' % self.trial)
            if could_load:
                print('Iteration:', self.step)
                print('=================================== Loading Succeeded ===================================')
                assert self.step == model_step, f'The latest step {model_step} and the input step {self.step} do not match.'
            else:
                print('=================================== No model to load ===================================')

            writer = tf.summary.FileWriter('./logs%d' % self.trial, sess.graph)
            print('Training Starts!')
            step = self.step

            t2 = time.time()
            if step == 0:
                print_time()
                save(self.saver, sess, self.checkpoint_dir, self.trial, step)

            while True:
                try:
                    inputa, labela, inputb, labelb, labela_gt, labelb_nousm = self.data_generator.generate_data(sess)

                    '''feed & fetch'''
                    feed_dict = {self.inputa: inputa, self.inputb: inputb, self.labela: labela, self.labelb: labelb, self.labelb_nousm: labelb_nousm}

                    if step == SECOND_ORDER_GRAD_ITER:
                        second_grad=sess.run(self.second_grad_on)
                        print('1st Order Gradients: ', second_grad)

                   # print(sess.run(self.all_us))
                    for update_ops in self.update_ops:
                        sess.run(update_ops)
                    #print(sess.run(self.all_us))
                    
                    sess.run(self.metatrain_op, feed_dict=feed_dict)
                    sess.run(self.d_op, feed_dict=feed_dict)

                    # You can run this if you want to use get_loss_weights
                    #sess.run(self.add_step)
                    step +=1

                    if step % PRINT_ITER == 0 or step == 1:
                        t1 = t2
                        t2 = time.time()

                        lossa_, lossb_, summary, M = sess.run([self.total_loss1, self.total_losses2[-1], self.summary_op, self.W], feed_dict=feed_dict)

                        print('Iteration:', step, '(Pre, Post) Loss:', lossa_, lossb_, 'Time: %.2f' % (t2 - t1))
                        print("Global Step: ", sess.run(self.global_step))
                        print("Step: ", step)
                        print("LW: ", sess.run(self.LW))
                        print("Max M: ", np.max(M))
                        print("Min M: ", np.min(M))
                        print("Mean M", np.mean(M))
                        writer.add_summary(summary, step)
                        writer.flush()


                    if step % SAVE_ITER == 0:
                        print_time()
                        save(self.saver, sess, self.checkpoint_dir, self.trial, step)

                    if step == self.META_ITER:
                        print('Done Training')
                        print_time()
                        break

                except KeyboardInterrupt:
                    print('***********KEY BOARD INTERRUPT *************')
                    print('Iteration:', step)
                    print_time()
                    save(self.saver, sess, self.checkpoint_dir, self.trial, step)
                    break

    def get_loss_weights(self):
        loss_weights = tf.ones(shape=[self.TASK_ITER]) * (1.0 / self.TASK_ITER)
        decay_rate = 1.0 / self.TASK_ITER / (10000 / 3)
        min_value= 0.03 / self.TASK_ITER

        loss_weights_pre = tf.maximum(loss_weights[:-1] - (tf.multiply(tf.to_float(self.global_step), decay_rate)), min_value)

        loss_weight_cur= tf.minimum(loss_weights[-1] + (tf.multiply(tf.to_float(self.global_step),(self.TASK_ITER- 1) * decay_rate)), 1.0 - ((self.TASK_ITER - 1) * min_value))
        loss_weights = tf.concat([[loss_weights_pre], [[loss_weight_cur]]], axis=1)
        return loss_weights